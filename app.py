import os
import json
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import bigquery
from langchain_google_vertexai import VertexAI
from collections import Counter
from app.utils import save_json,read_json
from collections import Counter

from app import (
    BatchParser,
    RequirementBuilder,
    MetadataEnricher,
    CategorizerRetriever,
    TestCaseGenerator,
    CoverageValidator,
    SemanticValidator,
    RegulationMapper
)


# -------------------------------
# Global Config
# -------------------------------
client = bigquery.Client()
PROJECT_ID = client.project
LOCATION = "us-central1"

os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION
LLM_MODEL = "gemini-2.5-pro"


# -------------------------------
# Layer 1 – Requirement Ingestion
# -------------------------------
from app.batch_parser import BatchParser

def parse_all_requirements(data_folder="data"):
    """
    Reads all supported files from the given folder,
    parses them, and returns a single combined list of requirements.
    """
    parser = BatchParser(data_folder=data_folder)
    results = parser.parse_batch()

    # Flatten results into one list
    all_reqs = []
    for _, reqs in results.items():
        all_reqs.extend(reqs)

    print(f"✅ Parsed {len(all_reqs)} total requirements from {len(results)} files.")
    return all_reqs

def run_requirement_builder(project_id, model="gemini-2.5-pro", batch_size=30, save_file="structured_requirements.json"):
    """
    Full pipeline to parse raw requirements and build structured requirements.
    
    Steps:
    1. Parse all raw requirements from 'data/' folder
    2. Use LLM to build structured requirements
    3. Save structured requirements locally as JSON
    """
    print("📂 Step 0: Parsing raw requirements...")
    all_req = parse_all_requirements()
    print(f"   → Parsed {len(all_req)} raw requirements")

    if not all_req:
        print("⚠️ No requirements found. Exiting pipeline.")
        return []

    print("🚀 Step 1: Building structured requirements...")
    builder = RequirementBuilder(model=model, project_id=project_id)

    structured_reqs = asyncio.run(builder.build_registry(all_req, batch_size=batch_size))
    print(f"   → Built {len(structured_reqs)} structured requirements")

    # Save locally
    if structured_reqs:
        save_json(structured_reqs, save_file)
        print(f"💾 Saved structured requirements → {save_file}")

    return structured_reqs
# -------------------------------
# Layer 3 – Metadata Enrichment
# -------------------------------
def load_requirements_from_bigquery(project_id, dataset_id="requirements_dataset", table_id="requirements"):
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    query = f"SELECT * FROM `{table_ref}`"
    rows = client.query(query).result()
    structured_reqs = [dict(row) for row in rows]

    for req in structured_reqs:
        if isinstance(req.get("metadata"), str):
            try:
                req["metadata"] = json.loads(req["metadata"])
            except Exception:
                req["metadata"] = {}

    print(f"Loaded {len(structured_reqs)} requirements from BigQuery.")
    return structured_reqs


def enrich_requirements(requirements, output_file="enriched_requirements.json"):
    enricher = MetadataEnricher()
    enriched_reqs = enricher.enrich(requirements)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enriched_reqs, f, indent=2,default=str)

    print(f"Enriched {len(enriched_reqs)} requirements → {output_file}")
    return enriched_reqs


# -------------------------------
# Layer 4 – Categorization & Retrieval
# -------------------------------
async def categorize_and_store_requirements(
    project_id,
    llm_model,
    enriched_file="enriched_requirements.json",
    embedding_model="text-embedding-005",
    dataset_id="requirements_dataset",
    table_id="requirements_categorized",
    batch_size=10,
):
    with open(enriched_file, "r", encoding="utf-8") as f:
        enriched_reqs = json.load(f)

    cr = CategorizerRetriever(
        project_id=project_id,
        embedding_model=embedding_model,
        classifier_model=llm_model
    )

    categorized_reqs = await cr.process_async(enriched_reqs, batch_size)
    cr.export_to_bq(categorized_reqs, dataset_id=dataset_id, table_id=table_id)

    print(f"Processed {len(categorized_reqs)} requirements → exported to {dataset_id}.{table_id}")
    return categorized_reqs


# -------------------------------
# Layer 5 – Test Case Generation
# -------------------------------
async def generate_and_store_test_cases(
    project_id,
    dataset_id="requirements_dataset",
    requirements_table="requirements_categorized",
    testcases_table="test_cases",
    save_local=True,
    local_file="test_cases.json",
    limit=None,
    batch_size=100,
    export_batch_size=200,
):
    client = bigquery.Client(project=project_id)
    query = f"""
        SELECT requirement_id, category, title, statement, priority, severity
        FROM `{project_id}.{dataset_id}.{requirements_table}`
    """
    rows = client.query(query).result()
    requirements = [dict(row) for row in rows]

    if limit:
        requirements = requirements[:limit]

    tcg = TestCaseGenerator(project_id)
    test_cases = await tcg.batch_generate_async(requirements, batch_size=batch_size)
    tcg.export_to_bq(test_cases, dataset_id=dataset_id, table_id=testcases_table, batch_size=export_batch_size)

    if save_local:
        with open(local_file, "w", encoding="utf-8") as f:
            json.dump(test_cases, f, indent=2)

    print(f"Exported {len(test_cases)} test cases to BigQuery → {dataset_id}.{testcases_table}")
    return test_cases


# -------------------------------
# Layer 6 – Coverage Validation
# -------------------------------
def build_traceability_matrix(project_id, dataset_id="requirements_dataset", table_name="traceability_matrix"):
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_name}"
    client.delete_table(table_ref, not_found_ok=True)
    print(f"Dropped old table {table_ref}")

    cv = CoverageValidator(project_id=project_id, dataset_id=dataset_id)
    trace_matrix = cv.build_traceability_matrix()
    df = pd.DataFrame(trace_matrix)
    return df


# -------------------------------
# Layer 6.5 – Semantic Validation
# -------------------------------
async def run_semantic_validation(
    project_id,
    dataset_id="requirements_dataset",
    requirements_table="requirements",
    testcases_table="test_cases",
    output_table="semantic_validation",
    limit=None,
    batch_size=10,
):
    client = bigquery.Client(project=project_id)

    req_query = f"SELECT DISTINCT requirement_id, statement FROM `{project_id}.{dataset_id}.{requirements_table}`"
    requirements = [dict(row) for row in client.query(req_query).result()]

    tc_query = f"SELECT DISTINCT test_id, requirement_id, title, description, steps, expected_result FROM `{project_id}.{dataset_id}.{testcases_table}`"
    test_cases = [dict(row) for row in client.query(tc_query).result()]

    if limit:
        requirements = requirements[:limit]
        test_cases = test_cases[:limit]

    sv = SemanticValidator(project_id=project_id)
    validated = await sv.validate_async(requirements, test_cases, batch_size=batch_size)
    sv.export_to_bq(validated, table_id=output_table)
    df = pd.DataFrame(validated)

    print(f"Exported {len(df)} validation results to {dataset_id}.{output_table}")
    return df


def show_regulation_coverage(enriched_reqs, total_reqs=None, logger=None):
    """
    Show compliance coverage stats.
    Calculates percentage of requirements mapped to each regulation.
    
    Args:
        enriched_reqs (list[dict]): List of enriched requirement dicts.
        total_reqs (int, optional): Total number of requirements in the project.
                                    If None, uses len(enriched_reqs).
        logger (Logger, optional): Logger instance. If None, prints to stdout.
    """
    all_regs = [reg for r in enriched_reqs for reg in r.get("regulation", [])]

    if all_regs:
        counts = Counter(all_regs)
        base_total = total_reqs if total_reqs is not None else len(enriched_reqs)

        msg_header = "\n📊 Regulation Coverage (relative to total requirements):"
        if logger:
            logger.info(msg_header)
        else:
            print(msg_header)

        for reg, count in counts.items():
            pct = (count / base_total) * 100 if base_total > 0 else 0
            msg = f"   - {reg}: {count}/{base_total} ({pct:.1f}%)"
            if logger:
                logger.info(msg)
            else:
                print(msg)
    else:
        msg = "\n⚠️ No regulations found in enriched requirements."
        if logger:
            logger.warning(msg)
        else:
            print(msg)


async def run_regulation_mapping(requirements, llm, batch_size=10):
    """
    Map regulations & obligations for given requirements.
    Updates each requirement with `regulation` and `obligations`.
    """
    # Initialize RegulationMapper with provided LLM
    mapper = RegulationMapper(regulation_file="regulations.yaml", llm=llm)

    texts = [req.get("statement", "") for req in requirements]
    mapped = await mapper.map_batch(texts, batch_size=batch_size)

    for req, m in zip(requirements, mapped):
        req["regulation"] = m["regulation"]      # column 1
        req["obligations"] = m["obligations"]    # column 2

    return requirements


def count_requirements_by_regulation(mapped_reqs):
    """
    Count how many requirements are mapped to each regulation.
    """
    all_regs = []
    for req in mapped_reqs:
        regs = req.get("regulation", [])
        if isinstance(regs, str):
            regs = [regs]
        all_regs.extend(regs)

    counts = Counter(all_regs)
    total_reqs = len(mapped_reqs)

    print("\n📊 Regulation Coverage (count of requirements mapped):")
    for reg, count in counts.most_common():
        pct = (count / total_reqs) * 100
        print(f" - {reg}: {count}/{total_reqs} ({pct:.1f}%)")

    return counts
# -------------------------------
# Main Entrypoint
# -------------------------------
if __name__ == "__main__":
      # ---------------- Run Pipeline ----------------
    print("🚀 Step 1: Building structured requirements...")
#     file_name="structured_requirements.json"
#     structured_reqs = run_requirement_builder(
#                 project_id=PROJECT_ID,
#                 model=LLM_MODEL,
#                 batch_size=50,
#                 save_file=file_name
#             )

#     structured_reqs=read_json(file_name)
    

    # Load structured requirements (already built & saved earlier)
    structured_reqs = read_json("structured_requirements.json")

    from langchain_google_vertexai import VertexAI

    llm = VertexAI(
        model_name="gemini-2.5-pro",
        temperature=0,
        project=PROJECT_ID,
        location=LOCATION,
    )
    regulation_mapped_reqs = asyncio.run(run_regulation_mapping(structured_reqs, llm=llm, batch_size=50))
    # Save enriched mapped_regs
    save_json(regulation_mapped_reqs, "regulation_mapped_reuirements.json")

    counts = count_requirements_by_regulation(regulation_mapped_reqs)
    print(counts)
    
    
#     print("🔍 Step 2: Enriching requirements with metadata...")
#     enriched_reqs = enrich_requirements(structured_reqs)
#     from app.compliance_validator import ComplianceValidator
#     from app.utils import save_json

#     # Load requirements (after enrichment step)
#     requirements = read_json("enriched_requirements.json")
#     # print(requirements[0])

#     validator = ComplianceValidator("regulations.yaml")
#     enriched_reqs, compliance_summary = validator.validate(requirements)

#     # Save enriched requirements
#     save_json(enriched_reqs, "compliance_enriched_requirements.json")

    
    # show_regulation_coverage(mapped_regs)


#     print("📊 Step 4: Categorizing requirements...")
#     categorized_reqs = asyncio.run(
#         categorize_and_store_requirements(
#             PROJECT_ID,
#             LLM_MODEL,
#             enriched_file="enriched_requirements.json",
#             batch_size=batch_size,
#         )
#     )

#     print("🧪 Step 5: Generating test cases...")
#     test_cases = asyncio.run(
#         generate_and_store_test_cases(
#             PROJECT_ID,
#             limit=top_limit,
#             batch_size=batch_size
#         )
#     )

#     print("🔗 Step 6: Building traceability matrix...")
#     df_trace = build_traceability_matrix(PROJECT_ID)
#     print(df_trace.head())

#     print("✅ Step 7: Running semantic validation...")
#     df_validated = asyncio.run(
#         run_semantic_validation(
#             PROJECT_ID,
#             limit=top_limit,
#             batch_size=batch_size
#         )
#     )
#     print(df_validated.head())

