# app/utils.py
import os
import json
import asyncio
import random
import logging
from typing import Any, List, Dict, Union
from google.cloud import bigquery


def get_logger(name: str, log_file="logs/automated_test_cases.log", level=logging.INFO) -> logging.Logger:
    """Create a logger with console + optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


class Utils:
    """Utility functions for JSON I/O, BigQuery access, and safe LLM calls."""

    def __init__(self, log_file: str = "logs/utils.log") -> None:
        self.logger = get_logger("Utils", log_file=log_file)

    # ---------------- JSON Utils ----------------
    def save_json(self, data: Any, file_name: str = "requirements.json") -> None:
        """Save Python object as JSON file."""
        try:
            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved {len(data)} records → {file_name}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON {file_name}: {e}")

    def read_json(self, file_path: str = "requirements.json") -> List[Dict[str, Any]]:
        """Load requirements from a JSON file."""
        if not os.path.exists(file_path):
            self.logger.warning(f"JSON file not found → {file_path}")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.logger.info(f"Loaded {len(data)} records from {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to read JSON {file_path}: {e}")
            return []
    def ensure_table(self, project_id, dataset_id, table_id, schema):
        """Ensure BigQuery table exists, else create it."""
        client = bigquery.Client(project=project_id)
        table_ref = f"{project_id}.{dataset_id}.{table_id}"

        try:
            client.get_table(table_ref)
            print(f"✅ Table exists: {table_ref}")
        except Exception:
            print(f"⚠️ Table not found. Creating {table_ref}...")
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table)
            print(f"✅ Created table: {table_ref}")

    def insert_rows(self, project_id, dataset_id, table_id, rows):
        """Insert rows into BigQuery table."""
        client = bigquery.Client(project=project_id)
        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        errors = client.insert_rows_json(table_ref, rows)
        if errors:
            print(f"❌ Errors inserting rows: {errors}")
        else:
            print(f"✅ Inserted {len(rows)} rows into {table_ref}")

    def load_requirements_to_bq(self, file_path, table_id, project_id=None, dataset_id="requirements_dataset"):
        """Load requirements JSON into BigQuery."""
        with open(file_path, "r", encoding="utf-8") as f:
            requirements = json.load(f)

        allowed_fields = {"requirement_id", "statement", "created_at"}
        filtered_reqs = [{k: v for k, v in req.items() if k in allowed_fields} for req in requirements]

        schema = [
            bigquery.SchemaField("requirement_id", "STRING"),
            bigquery.SchemaField("statement", "STRING"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
        ]

        self.ensure_table(project_id, dataset_id, table_id, schema)
        self.insert_rows(project_id, dataset_id, table_id, filtered_reqs)

    def load_testcases_to_bq(self, file_path, table_id, project_id=None, dataset_id="requirements_dataset"):
        """Load test cases JSON into BigQuery."""
        with open(file_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        schema = [
            bigquery.SchemaField("test_id", "STRING"),
            bigquery.SchemaField("requirement_id", "STRING"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("description", "STRING"),
            bigquery.SchemaField("preconditions", "STRING", mode="REPEATED"),
            bigquery.SchemaField("steps", "STRING", mode="REPEATED"),
            bigquery.SchemaField("test_data", "STRING"),
            bigquery.SchemaField("expected_result", "STRING", mode="REPEATED"),
            bigquery.SchemaField("postconditions", "STRING", mode="REPEATED"),
            bigquery.SchemaField("priority", "STRING"),
            bigquery.SchemaField("severity", "STRING"),
            bigquery.SchemaField("type", "STRING"),
            bigquery.SchemaField("execution_status", "STRING"),
            bigquery.SchemaField("owner", "STRING"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
        ]

        # Convert dict → JSON string for test_data
        for tc in test_cases:
            if isinstance(tc.get("test_data"), dict):
                tc["test_data"] = json.dumps(tc["test_data"], ensure_ascii=False)

        self.ensure_table(project_id, dataset_id, table_id, schema)
        self.insert_rows(project_id, dataset_id, table_id, test_cases)
    # ---------------- BigQuery Utils ----------------
    def read_from_bigquery(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str = "requirements",
        limit: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Read rows from a BigQuery table."""
        client = bigquery.Client(project=project_id)
        table_ref = f"{project_id}.{dataset_id}.{table_id}"

        query = f"SELECT * FROM `{table_ref}`"
        if limit:
            query += f" LIMIT {limit}"

        try:
            self.logger.info(f"Querying BigQuery: {table_ref}")
            query_job = client.query(query)
            rows = [dict(row) for row in query_job.result()]
            self.logger.info(f"Loaded {len(rows)} rows from {table_ref}")
            return rows
        except Exception as e:
            self.logger.error(f"Failed to load from BigQuery {table_ref}: {e}")
            return []

    # ---------------- LLM Utils ----------------
    async def safe_llm_call(
        self,
        llm: Any,
        prompt: str,
        retries: int = 3,
        timeout: int = 60,
        min_backoff: float = 1.0,
        max_backoff: float = 8.0,
    ) -> Union[dict, object]:
        """Safely call an LLM with retries, exponential backoff, and timeout."""
        for attempt in range(1, retries + 1):
            try:
                self.logger.debug(f"Attempt {attempt}/{retries} for LLM call")
                response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=timeout)
                if response:
                    return response
                self.logger.warning(f"Empty response (attempt {attempt})")
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout on attempt {attempt}")
            except Exception as e:
                self.logger.error(f"Error on attempt {attempt}: {e}")

            if attempt < retries:
                wait_time = min(max_backoff, min_backoff * (2 ** (attempt - 1)))
                wait_time *= random.uniform(0.8, 1.3)  # jitter
                self.logger.info(f"Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)

        self.logger.error("All retries failed, returning fallback response.")
        return {
            "success": False,
            "error": "LLM failed after retries",
            "prompt": prompt,
            "response": None,
        }

    async def safe_llm_batch(
        self,
        llm: Any,
        prompts: List[str],
        retries: int = 3,
        timeout: int = 60,
        batch_size: int = 10,
        max_concurrent: int = 5,
    ) -> List[Any]:
        """Process multiple prompts with safe LLM calls (batched + rate-limited)."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_call(prompt: str):
            async with semaphore:
                return await self.safe_llm_call(llm, prompt, retries=retries, timeout=timeout)

        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            tasks = [limited_call(p) for p in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(batch_results)

        return results
