#!/usr/bin/env python
# coding: utf-8

import os
import json
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import bigquery
from langchain_google_vertexai import VertexAI

from app import (
    BatchParser,
    RequirementBuilder,
    MetadataEnricher,
    CategorizerRetriever,
    TestCaseGenerator,
    CoverageValidator,
    SemanticValidator,
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
def parse_and_export_requirements():
    parser = BatchParser(data_folder="data")
    parsed_results = parser.parse_batch()
    parser.export_results(
        results=parsed_results,
        project_id=PROJECT_ID,
        dataset_id="requirements_dataset",
        table_id="raw_chunks"
    )
    print("Parsed and exported requirements.")


# -------------------------------
# Layer 2 – Requirement Registry
# -------------------------------
async def build_registry():
    builder = RequirementBuilder(model=LLM_MODEL, project_id=PROJECT_ID)

    with open("requirements.json", "r", encoding="utf-8") as f:
        requirements = json.load(f)

    structured_reqs = await builder.build_registry(requirements, batch_size=100)

    with open("structured_requirements.json", "w", encoding="utf-8") as f:
        json.dump(structured_reqs, f, indent=2)

    print(f"Saved {len(structured_reqs)} structured requirements → structured_requirements.json")

    try:
        builder.export_to_bq(
            structured_reqs,
            dataset_id="requirements_dataset",
            table_id="requirements",
            batch_size=100
        )
    except Exception as e:
        print(f"BigQuery export failed: {e} (saved locally instead)")

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
        json.dump(enriched_reqs, f, indent=2)

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


# -------------------------------
# Main Entrypoint
# -------------------------------
if __name__ == "__main__":
    # Toggle this flag for testing vs full run
    testing = True  

    # Default values
    top_limit = None
    batch_size = 100

    if testing:
        top_limit = 10    # only take 10 requirements/testcases
        batch_size = 10   # smaller batches to avoid quota issues

    # ---------------- Run Pipeline ----------------
    # parse_and_export_requirements()
    structured_reqs=asyncio.run(build_registry())
    
    reqs = load_requirements_from_bigquery(PROJECT_ID)
    enriched_reqs=enrich_requirements(reqs)

    categorized_reqs=asyncio.run(
        categorize_and_store_requirements(PROJECT_ID, LLM_MODEL, enriched_file="enriched_requirements.json",batch_size=batch_size,)
    )
    
    test_cases=asyncio.run(
        generate_and_store_test_cases(PROJECT_ID, limit=top_limit, batch_size=batch_size)
    )

    df_trace = build_traceability_matrix(PROJECT_ID)
    print(df_trace.head())

    df_validated = asyncio.run(
        run_semantic_validation(PROJECT_ID, limit=top_limit, batch_size=batch_size)
    )
    print(df_validated.head())

