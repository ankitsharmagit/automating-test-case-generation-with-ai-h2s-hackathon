import json
import datetime
import re
import uuid
from google.cloud import bigquery
from langchain_google_vertexai import VertexAI
from tqdm import tqdm

class SemanticValidator:
    def __init__(self, project_id, dataset_id="requirements_dataset", model="gemini-2.5-pro"):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.llm = VertexAI(model_name=model, temperature=0)

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
        """Try to safely parse JSON from model response"""
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

    def validate(self, requirements, test_cases):
        """Run semantic validation for all test cases"""
        validated = []
        for tc in tqdm(test_cases, desc="Semantic validating test cases"):
            req = next((r for r in requirements if r["requirement_id"] == tc["requirement_id"]), None)
            if not req:
                continue

            prompt = self._make_prompt(req, tc)

            try:
                response = self.llm.invoke(prompt)
                raw_text = response if isinstance(response, str) else getattr(response, "content", str(response))
                result = self._safe_parse_json(raw_text)
            except Exception as e:
                print(f"⚠️ Validation failed for {tc['test_id']}: {e}")
                result = {"matches": None, "confidence": 0, "reason": "LLM error"}

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

    def export_to_bq(self, validated, table_id="semantic_validation"):
        """Save results into a new BigQuery table"""
        table_ref = f"{self.project_id}.{self.dataset_id}.{table_id}"

        schema = [
            bigquery.SchemaField("validation_id", "STRING"),
            bigquery.SchemaField("test_id", "STRING"),
            bigquery.SchemaField("requirement_id", "STRING"),
            bigquery.SchemaField("semantic_matches", "BOOL"),
            bigquery.SchemaField("semantic_confidence", "FLOAT"),
            bigquery.SchemaField("semantic_reason", "STRING"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
        ]

        try:
            self.client.get_table(table_ref)
            print(f"✅ Table {table_ref} exists.")
        except Exception:
            table = bigquery.Table(table_ref, schema=schema)
            self.client.create_table(table)
            print(f"✅ Created table {table_ref}")

        errors = self.client.insert_rows_json(table_ref, validated)
        if errors:
            print(f"❌ Errors inserting rows: {errors}")
        else:
            print(f"✅ Inserted {len(validated)} rows into {table_ref}")
