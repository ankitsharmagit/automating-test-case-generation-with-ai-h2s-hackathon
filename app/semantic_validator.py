import asyncio
import json
import datetime
import re
import uuid
import random
from google.cloud import bigquery
from langchain_google_vertexai import VertexAI
from tqdm import tqdm


# ------------------------------
# Async Safe Batch with Retry
# ------------------------------
async def safe_llm_batch_async(llm, prompts, timeout=90, max_concurrent=3, max_retries=5):
    """Run multiple LLM calls concurrently with throttling + exponential backoff."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def call_with_retry(prompt):
        for attempt in range(max_retries):
            try:
                async with semaphore:
                    return await llm.ainvoke(prompt)
            except Exception as e:
                if "429" in str(e) or "Resource exhausted" in str(e):
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"⚠️ Rate limit hit. Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    return {"matches": False, "confidence": 0, "reason": f"Error: {e}"}
        return {"matches": False, "confidence": 0, "reason": "Max retries exceeded"}

    tasks = [call_with_retry(p) for p in prompts]
    try:
        return await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
    except asyncio.TimeoutError:
        print("⏳ Timeout reached, skipping this batch.")
        return ["{}" for _ in prompts]


# ------------------------------
# Semantic Validator Class
# ------------------------------
class SemanticValidator:
    def __init__(self, project_id, dataset_id="requirements_dataset",
                 model="gemini-2.5-pro", location="us-central1"):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.llm = VertexAI(model_name=model, temperature=0,
                            project=project_id, location=location)

    def _make_prompt(self, requirement, test_case):
        return f"""
        You are validating whether a test case matches its requirement.

        Requirement:
        ID: {requirement.get('requirement_id', '')}
        Statement: {requirement.get('statement', '')}

        Test Case:
        ID: {test_case.get('test_id', '')}
        Title: {test_case.get('title', '')}
        Description: {test_case.get('description', '')}
        Steps: {test_case.get('steps', [])}
        Expected Results: {test_case.get('expected_result', [])}

        Respond ONLY with strict JSON in this format:
        {{
          "matches": true/false,
          "confidence": 0-100,
          "reason": "short explanation"
        }}
        """

    def _safe_parse_json(self, raw_text):
        """Try to safely parse JSON from model response."""
        try:
            return json.loads(raw_text)
        except Exception:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    return {"matches": False, "confidence": 0, "reason": "Invalid JSON returned"}
            return {"matches": False, "confidence": 0, "reason": "No JSON returned"}

    async def validate_async(self, requirements, test_cases, batch_size=10):
        """Run semantic validation asynchronously in batches."""
        validated, prompts, tc_meta = [], [], []

        # Build prompts
        for tc in test_cases:
            req = next((r for r in requirements if r["requirement_id"] == tc["requirement_id"]), None)
            if not req:
                continue
            prompts.append(self._make_prompt(req, tc))
            tc_meta.append(tc)

        # Run in async batches
        for i in tqdm(range(0, len(prompts), batch_size), desc="Semantic validating test cases"):
            batch_prompts = prompts[i:i+batch_size]
            batch_meta = tc_meta[i:i+batch_size]

            responses = await safe_llm_batch_async(self.llm, batch_prompts)

            for tc, resp in zip(batch_meta, responses):
                raw_text = str(resp) if isinstance(resp, str) else getattr(resp, "content", str(resp))
                result = self._safe_parse_json(raw_text)

                validated.append({
                    "validation_id": str(uuid.uuid4()),
                    "test_id": tc["test_id"],
                    "requirement_id": tc["requirement_id"],
                    "semantic_matches": result.get("matches"),
                    "semantic_confidence": result.get("confidence"),
                    "semantic_reason": result.get("reason"),
                    "created_at": datetime.datetime.utcnow().isoformat()
                })

        return validated

    def export_to_bq(self, validated, table_id="semantic_validation", batch_size=100):
        """Save results into a BigQuery table with auto schema check and dataset creation."""
        table_ref = f"{self.project_id}.{self.dataset_id}.{table_id}"

        expected_schema = [
            bigquery.SchemaField("validation_id", "STRING"),
            bigquery.SchemaField("test_id", "STRING"),
            bigquery.SchemaField("requirement_id", "STRING"),
            bigquery.SchemaField("semantic_matches", "BOOL"),
            bigquery.SchemaField("semantic_confidence", "FLOAT"),
            bigquery.SchemaField("semantic_reason", "STRING"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
        ]

        # ✅ Ensure dataset exists
        dataset_ref = bigquery.Dataset(f"{self.project_id}.{self.dataset_id}")
        try:
            self.client.get_dataset(dataset_ref)
        except Exception:
            self.client.create_dataset(dataset_ref, exists_ok=True)
            print(f"✅ Created dataset {self.dataset_id}")

        # ✅ Ensure table schema matches
        recreate = False
        try:
            table = self.client.get_table(table_ref)
            existing_fields = {f.name for f in table.schema}
            expected_fields = {f.name for f in expected_schema}
            if existing_fields != expected_fields:
                print(f"⚠️ Schema mismatch. Recreating table {table_ref}.")
                self.client.delete_table(table_ref, not_found_ok=True)
                recreate = True
        except Exception:
            recreate = True

        if recreate:
            table = bigquery.Table(table_ref, schema=expected_schema)
            self.client.create_table(table)
            print(f"✅ Created table {table_ref} with expected schema.")

        # ✅ Insert in batches
        total_inserted = 0
        for i in range(0, len(validated), batch_size):
            batch = validated[i:i+batch_size]
            errors = self.client.insert_rows_json(table_ref, batch)
            if errors:
                print(f"❌ Errors inserting batch {i//batch_size+1}: {errors}")
            else:
                total_inserted += len(batch)
                print(f"✅ Inserted batch {i//batch_size+1} ({len(batch)} rows).")

        print(f"Finished inserting {total_inserted} rows into {table_ref}")

