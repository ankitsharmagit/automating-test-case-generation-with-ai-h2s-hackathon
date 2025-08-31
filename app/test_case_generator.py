import asyncio
import datetime
import uuid
import json
import re
from tqdm import tqdm
from google.cloud import bigquery
from langchain_google_vertexai import VertexAI


async def safe_llm_batch_async(llm, prompts, timeout=90):
    """Run multiple LLM calls concurrently with timeout handling."""
    tasks = [llm.ainvoke(prompt) for prompt in prompts]
    try:
        return await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        print("Timeout reached, skipping this batch.")
        return ["[]" for _ in prompts]


class TestCaseGenerator:
    def __init__(self, project_id, location="us-central1"):
        self.project_id = project_id
        self.location = location
        self.llm = VertexAI(
            model_name="gemini-2.5-pro",
            temperature=0,
            project=project_id,
            location=location
        )

    def _make_prompt(self, requirement):
        return f"""
        You are a healthcare QA test designer.
        Convert the following requirement into 3 test cases:
        - Positive (happy path)
        - Negative (invalid input/failure)
        - Edge (boundary condition)

        Requirement: "{requirement['statement']}"

        For each test case, return JSON with fields:
        - test_id
        - requirement_id
        - title
        - description
        - preconditions (list of strings)
        - steps (list of strings)
        - test_data (JSON object)
        - expected_result (list of strings)
        - postconditions (list of strings)
        - priority (P1–P3)
        - severity (Critical/Major/Minor/Cosmetic)
        - type (Functional, Security, Performance, etc.)
        - execution_status (default: "Not Executed")
        - owner (default: "QA Team")
        - created_at (timestamp)

        Return ONLY valid JSON array of 3 objects.
        """

    def _extract_json(self, text):
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return []
        return []

    async def batch_generate_async(self, requirements, batch_size=5):
        """Generate test cases asynchronously in batches of requirements."""
        all_cases = []
        prompts = [self._make_prompt(req) for req in requirements]

        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Test Cases"):
            batch_prompts = prompts[i:i+batch_size]
            batch_reqs = requirements[i:i+batch_size]

            responses = await safe_llm_batch_async(self.llm, batch_prompts)

            for resp, req in zip(responses, batch_reqs):
                raw_text = str(resp) if isinstance(resp, str) else getattr(resp, "content", str(resp))
                cases = self._extract_json(raw_text)

                for case in cases:
                    all_cases.append({
                        "test_id": case.get("test_id", f"TC-{uuid.uuid4()}"),
                        "requirement_id": req["requirement_id"],
                        "title": case.get("title", "Untitled Test Case"),
                        "description": case.get("description", ""),
                        "preconditions": case.get("preconditions", []),
                        "steps": case.get("steps", []),
                        "test_data": case.get("test_data", {}),
                        "expected_result": case.get("expected_result", []),
                        "postconditions": case.get("postconditions", []),
                        "priority": case.get("priority", req.get("priority", "P3")),
                        "severity": case.get("severity", req.get("severity", "Minor")),
                        "type": case.get("type", req.get("category", "Functional")),
                        "execution_status": "Not Executed",
                        "owner": "QA Team",
                        "created_at": datetime.datetime.utcnow().isoformat()
                    })

        return all_cases

    def export_to_bq(self, test_cases, dataset_id="requirements_dataset", table_id="test_cases", batch_size=100):
        """Export test cases into BigQuery in safe batches."""
        if not test_cases:
            print("No test cases to export")
            return

        client = bigquery.Client(project=self.project_id)
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"

        schema = [
            bigquery.SchemaField("test_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("requirement_id", "STRING"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("description", "STRING"),
            bigquery.SchemaField("preconditions", "STRING", mode="REPEATED"),
            bigquery.SchemaField("steps", "STRING", mode="REPEATED"),
            bigquery.SchemaField("test_data", "STRING"),   # store JSON as string
            bigquery.SchemaField("expected_result", "STRING", mode="REPEATED"),
            bigquery.SchemaField("postconditions", "STRING", mode="REPEATED"),
            bigquery.SchemaField("priority", "STRING"),
            bigquery.SchemaField("severity", "STRING"),
            bigquery.SchemaField("type", "STRING"),
            bigquery.SchemaField("execution_status", "STRING"),
            bigquery.SchemaField("owner", "STRING"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
        ]

        # Ensure table exists
        try:
            client.get_table(table_ref)
            print(f"Table {table_ref} exists.")
        except Exception:
            print(f"Table {table_ref} not found. Creating...")
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table)
            print(f"Created table {table_ref}")

        # Prepare rows
        all_rows = []
        for tc in test_cases:
            row = tc.copy()
            if isinstance(row.get("test_data"), dict):
                row["test_data"] = json.dumps(row["test_data"], ensure_ascii=False)
            all_rows.append(row)

        # Insert in batches
        total_inserted = 0
        for i in range(0, len(all_rows), batch_size):
            batch = all_rows[i:i+batch_size]
            errors = client.insert_rows_json(table_ref, batch)
            if errors:
                print(f"Errors inserting batch {i//batch_size + 1}: {errors}")
            else:
                total_inserted += len(batch)
                print(f"Inserted batch {i//batch_size + 1} ({len(batch)} rows).")

        print(f"Finished inserting {total_inserted} test cases into {table_ref}")
