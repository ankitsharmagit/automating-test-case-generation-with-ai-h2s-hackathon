import asyncio
import json
from tqdm import tqdm
from google.cloud import bigquery
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI


async def safe_llm_batch_async(llm, prompts, timeout=60):
    """Run multiple LLM calls concurrently with timeout handling."""
    tasks = [llm.ainvoke(prompt) for prompt in prompts]
    try:
        return await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        print("Timeout reached, skipping batch.")
        return ["Uncategorized" for _ in prompts]


class CategorizerRetriever:
    def __init__(self, project_id, location="us-central1",
                 embedding_model="text-embedding-005",
                 classifier_model="gemini-2.5-pro"):
        self.project_id = project_id
        self.location = location
        self.embedder = VertexAIEmbeddings(model=embedding_model,
                                           project=project_id,
                                           location=location)
        self.classifier = VertexAI(model_name=classifier_model,
                                   temperature=0,
                                   project=project_id,
                                   location=location)

    def _make_classify_prompt(self, text: str) -> str:
        return f"""
        You are a strict requirements classifier.

        Task:
        Categorize the following requirement into exactly ONE of these categories:
        - Functional
        - Security
        - Performance
        - Usability
        - Compliance
        - Reliability
        - Others

        Requirement:
        "{text}"

        Rules:
        - Return ONLY the category name (no explanations, no extra text).
        - If unsure, choose "Others".
        """



    async def process_async(self, enriched_reqs, batch_size=10):
        """Async classify + generate embeddings for enriched requirements."""
        categorized = []

        # --- Step 1: Build prompts ---
        prompts = [self._make_classify_prompt(req.get("statement", ""))
                   for req in enriched_reqs]

        # --- Step 2: Run classification in async batches ---
        categories = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Classifying"):
            batch_prompts = prompts[i:i+batch_size]
            responses = await safe_llm_batch_async(self.classifier, batch_prompts)
            cleaned = []
            for resp in responses:
                if isinstance(resp, str):
                    cleaned.append(resp.strip())
                else:
                    cleaned.append(getattr(resp, "content", "").strip())
            categories.extend(cleaned)

        # --- Step 3: Run embeddings concurrently (still sync in SDK) ---
        # Note: VertexAIEmbeddings.embed_query() is sync-only, so we run in executor
        loop = asyncio.get_event_loop()
        embedding_tasks = [
            loop.run_in_executor(None, self.embedder.embed_query, req.get("statement", ""))
            if req.get("statement") else loop.run_in_executor(None, lambda: [])
            for req in enriched_reqs
        ]
        embeddings = await asyncio.gather(*embedding_tasks)

        # --- Step 4: Attach results ---
        for req, cat, emb in zip(enriched_reqs, categories, embeddings):
            req["category"] = cat if cat else "Uncategorized"
            req["embedding"] = emb if emb else []
            categorized.append(req)

        return categorized

    def _infer_bq_schema(self, sample_row):
        """Infer BigQuery schema dynamically from a sample requirement dict."""
        schema = []
        for key, val in sample_row.items():
            if key == "embedding":
                schema.append(bigquery.SchemaField(key, "FLOAT64", mode="REPEATED"))
            elif isinstance(val, list):
                schema.append(bigquery.SchemaField(key, "STRING", mode="REPEATED"))
            elif isinstance(val, dict):
                schema.append(bigquery.SchemaField(key, "JSON"))
            elif isinstance(val, str):
                schema.append(bigquery.SchemaField(key, "STRING"))
            elif isinstance(val, (int, float)):
                schema.append(bigquery.SchemaField(key, "FLOAT64"))
            else:
                schema.append(bigquery.SchemaField(key, "STRING"))  # fallback
        return schema

    def export_to_bq(self, categorized_reqs, dataset_id="requirements_dataset", table_id="requirements"):
        """Export categorized requirements into BigQuery with dynamic schema inference (dicts → strings)."""
        if not categorized_reqs:
            print("No data to export")
            return

        client = bigquery.Client(project=self.project_id)
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"

        # Infer schema from first row
        schema = self._infer_bq_schema(categorized_reqs[0])

        # Ensure dataset exists
        try:
            client.get_dataset(dataset_id)
        except Exception:
            print(f"Dataset {dataset_id} not found. Creating...")
            dataset = bigquery.Dataset(f"{self.project_id}.{dataset_id}")
            dataset.location = self.location
            client.create_dataset(dataset, exists_ok=True)
            print(f"Created dataset {dataset_id}")

        # Ensure table exists
        try:
            client.get_table(table_ref)
            print(f"Table {table_ref} exists.")
        except Exception:
            print(f"Table {table_ref} not found. Creating...")
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table)
            print(f"Created table {table_ref}")

        # Stringify dicts before insert
        rows = []
        for req in categorized_reqs:
            row = req.copy()
            for k, v in row.items():
                if isinstance(v, dict):
                    row[k] = json.dumps(v, ensure_ascii=False)
            rows.append(row)

        # Insert rows
        errors = client.insert_rows_json(table_ref, rows)
        if errors:
            print(f"Errors inserting rows: {errors}")
        else:
            print(f"Inserted {len(rows)} categorized requirements into {table_ref}")
