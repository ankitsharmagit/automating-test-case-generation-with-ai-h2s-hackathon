# app.py
import os
import json
import asyncio
import pandas as pd
from google.cloud import bigquery
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings

from app.utils import Utils, get_logger
from app import (
    BatchParser,
    RequirementBuilder,
    MetadataEnricher,
    CategorizerRetriever,
    TestCaseGenerator,
    CoverageValidator,
    SemanticValidator,
    RegulationMapper,
)

# -------------------------
# Initialize Utils + Logger
# -------------------------
utils = Utils()
logger = get_logger("Pipeline", log_file="logs/pipeline.log")


# -------------------------------
# Layer 1 – Requirement Ingestion
# -------------------------------
def parse_all_requirements(data_folder="data"):
    parser = BatchParser(data_folder=data_folder)
    results = parser.parse_batch()
    all_reqs = [req for _, reqs in results.items() for req in reqs]
    logger.info(f"Parsed {len(all_reqs)} total requirements from {len(results)} files.")
    return all_reqs


def run_requirement_builder(
    project_id,
    llm_name,
    output_file,
    batch_size=30,
):
    logger.info("Step 0: Parsing raw requirements...")
    all_req = parse_all_requirements()

    if not all_req:
        logger.warning("No requirements found. Exiting pipeline.")
        return []

    logger.info("Step 1: Building structured requirements...")
    builder = RequirementBuilder(model=llm_name, project_id=project_id)
    structured_reqs = asyncio.run(builder.build_registry(all_req, batch_size=batch_size))
    logger.info(f"Built {len(structured_reqs)} structured requirements")

    if structured_reqs:
        utils.save_json(structured_reqs, output_file)
    return structured_reqs


# -------------------------------
# Layer 3 – Metadata Enrichment
# -------------------------------
# def load_requirements_from_bigquery(project_id, dataset_id, table_id):
#     structured_reqs = utils.read_from_bigquery(project_id, dataset_id, table_id)
#     for req in structured_reqs:
#         if isinstance(req.get("metadata"), str):
#             try:
#                 req["metadata"] = json.loads(req["metadata"])
#             except Exception:
#                 req["metadata"] = {}
#     logger.info(f"Loaded {len(structured_reqs)} requirements from BigQuery.")
#     return structured_reqs


def enrich_requirements(requirements, output_file):
    enricher = MetadataEnricher()
    enriched_reqs = enricher.enrich(requirements)
    if enriched_reqs:
        utils.save_json(enriched_reqs, output_file)
    return enriched_reqs


# -------------------------------
# Layer 4 – Categorization & Retrieval
# -------------------------------
async def categorize_and_store_requirements(
    project_id,
    llm_model,
    enriched_file,
    embedding_model,
    dataset_id,
    table_id,
    batch_size=10,
):
    enriched_reqs = utils.read_json(enriched_file)

    cr = CategorizerRetriever(
        project_id=project_id,
        embedding_model=embedding_model,
        classifier_model=llm_model,
    )

    categorized_reqs = await cr.process_async(enriched_reqs, batch_size)
    cr.export_to_bq(categorized_reqs, dataset_id=dataset_id, table_id=table_id)
    return categorized_reqs


# -------------------------------
# Layer 5 – Test Case Generation
# -------------------------------
async def generate_and_store_test_cases(
    requirements,
    model,
    output_file,
    batch_size=100,
):
    # # client = bigquery.Client(project=project_id)
    # query = f"""
    #     SELECT requirement_id, category, title, statement, priority, severity
    #     FROM `{project_id}.{dataset_id}.{requirements_table}`
    # """
    # rows = client.query(query).result()
    # requirements = [dict(row) for row in rows]

    tcg = TestCaseGenerator(model=model)
    test_cases = await tcg.batch_generate_async(requirements, batch_size=batch_size)
    
    # tcg.export_to_bq(
    #     test_cases,
    #     dataset_id=dataset_id,
    #     table_id=testcases_table,
    #     batch_size=export_batch_size,
    # )

    if test_cases:
        utils.save_json(test_cases, output_file)
    return test_cases


# -------------------------------
# Regulation Mapping
# -------------------------------
async def run_regulation_mapping(requirements, model, output_file, batch_size=10,):
    mapper = RegulationMapper(regulation_file="regulations.yaml", model=model)
    texts = [req.get("statement", "") for req in requirements]
    mapped = await mapper.map_batch(texts, batch_size=batch_size)

    for req, m in zip(requirements, mapped):
        req["regulation"] = m.get("regulation", [])
        req["obligations"] = m.get("obligations", [])
    if requirements :
        utils.save_json(requirements, output_file)
        
    return requirements


def count_requirements_by_regulation(mapped_reqs):
    counts = {}
    for req in mapped_reqs:
        regs = req.get("regulation", [])
        if isinstance(regs, str):
            regs = [regs]
        for reg in regs:
            counts[reg] = counts.get(reg, 0) + 1
    return counts


def show_regulation_coverage(mapped_reqs, logger=None):
    counts = count_requirements_by_regulation(mapped_reqs)
    total = len(mapped_reqs)
    if logger:
        logger.info("Regulation Coverage Summary:")
        for reg, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = (count / total) * 100 if total else 0
            logger.info(f" - {reg}: {count}/{total} ({pct:.1f}%)")
    return counts


# -------------------------------
# Main Entrypoint
# -------------------------------
if __name__ == "__main__":
    
    # -------------------------
    # Global Config & Constants
    # -------------------------
    client = bigquery.Client()
    PROJECT_ID = client.project
    LOCATION = "us-central1"
    DATASET_ID = "requirements_dataset"

    # -------------------------
    # Model Configs
    # -------------------------
    LLM_MODEL = "gemini-2.5-pro"
    EMBEDDING_MODEL = "text-embedding-005"

    # -------------------------
    # Paths
    # -------------------------
    RAW_REQ_FILE = "requirements.json"
    STRUCTURED_REQ_FILE = "structured_requirements.json"
    ENRICHED_REQ_FILE = "enriched_requirements.json"
    MAPPED_REQ_FILE = "regulation_mapped_requirements.json"
    TEST_CASES_FILE = "test_cases.json"

    # -------------------------
    # BQ Table Names
    # -------------------------
    REQ_TABLE = "requirements"
    TC_TABLE = "test_cases"
    TRACE_TABLE = "traceability_matrix"
    SEMANTIC_TABLE = "semantic_validation"

    # -------------------------
    # Environment Vars
    # -------------------------
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION

    # LLM + embeddings
    # llm = VertexAI(model_name=LLM_MODEL, temperature=0, project=PROJECT_ID, location=LOCATION)
    # embedding_model = VertexAIEmbeddings(model=EMBEDDING_MODEL, project=PROJECT_ID, location=LOCATION)

    # Step 1: Build structured requirements
    structured_reqs = run_requirement_builder(PROJECT_ID,
                                              LLM_MODEL, 
                                              output_file=STRUCTURED_REQ_FILE,
                                              batch_size=100, )

    # Step 2: Regulation Mapping
    structured_reqs=utils.read_json(STRUCTURED_REQ_FILE)
    regulation_mapped_reqs = asyncio.run(run_regulation_mapping(structured_reqs,                             
                                                                model=LLM_MODEL,
                                                                output_file=MAPPED_REQ_FILE,
                                                                batch_size=100,))
    show_regulation_coverage(regulation_mapped_reqs, logger)

    # Step 3: Metadata Enrichment
    enriched_reqs = enrich_requirements(regulation_mapped_reqs, output_file=ENRICHED_REQ_FILE)
    

    # Step 4: Test Case Generation
    enriched_reqs=utils.read_json(ENRICHED_REQ_FILE)
    limit=5
    if limit:
        enriched_reqs = enriched_reqs[:limit]
        
        
    logger.info(f"{len(enriched_reqs)} requirements loaded for test case generation")
    test_cases = asyncio.run(generate_and_store_test_cases(enriched_reqs, 
                                                            model=LLM_MODEL,
                                                            output_file=TEST_CASES_FILE,
                                                            batch_size=100))

    # Step 5: Load Requirements and Testcases to BigQuery
    utils.load_requirements_to_bq(RAW_REQ_FILE, REQ_TABLE, project_id=PROJECT_ID, dataset_id=DATASET_ID)
    utils.load_testcases_to_bq(TEST_CASES_FILE, TC_TABLE, project_id=PROJECT_ID, dataset_id=DATASET_ID)


    # Step 6: Traceability Matrix
    cv = CoverageValidator(project_id=PROJECT_ID, dataset_id=DATASET_ID)
    trace_matrix = cv.build_traceability_matrix(
        requirements_table=REQ_TABLE,
        testcases_table=TC_TABLE,
        output_table=TRACE_TABLE,
    )
    df_trace = pd.DataFrame(trace_matrix)
    print(df_trace.head())

    # Step 7: Semantic Validation
    enriched_reqs = utils.read_json(ENRICHED_REQ_FILE)
    test_cases = utils.read_json(TEST_CASES_FILE)
    sv = SemanticValidator(project_id=PROJECT_ID)
    validated = asyncio.run(
        sv.validate_async(enriched_reqs, test_cases, batch_size=20)
    )
    sv.export_to_bq(validated, table_id=SEMANTIC_TABLE)
    df_validated = pd.DataFrame(validated)
    print(df_validated.head())
