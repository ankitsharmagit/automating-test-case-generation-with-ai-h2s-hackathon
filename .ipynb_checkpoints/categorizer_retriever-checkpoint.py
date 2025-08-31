import json
from tqdm import tqdm
from google.cloud import bigquery
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI


class CategorizerRetriever:
    def __init__(self, project_id, location="us-central1",embedding_model="text-embedding-005",classifier_model="gemini-2.5-pro"):
        self.project_id = project_id
        self.location = location
        self.embedder = VertexAIEmbeddings(model=embedding_model, project=project_id, location=location)
        self.classifier = VertexAI(model_name=classifier_model, temperature=0, project=project_id, location=location)

    def _classify_category(self, text):
        """Use LLM to classify requirement into a standard category."""
        prompt = f"""
        You are a requirements classifier. Categorize the following requirement into one of:
        [Functional, Security, Performance, Usability, Compliance, Reliability].

        Requirement: "{text}"

        Return ONLY the category name.
        """
        resp = self.classifier.invoke(prompt)
        return resp.strip()

    def process(self, enriched_reqs):
        """Classify + generate embeddings for enriched requirements."""
        categorized = []
        for req in tqdm(enriched_reqs, desc="Categorizing + Embedding"):
            statement = req.get("statement", "")

            # Step 1: Category
            category = self._classify_category(statement) if statement else "Uncategorized"
            req["category"] = category

            # Step 2: Embedding
            if statement:
                emb = self.embedder.embed_query(statement)
                req["embedding"] = emb
            else:
                req["embedding"] = []

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
            print("⚠️ No data to export")
            return

        client = bigquery.Client(project=self.project_id)
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"

        # Infer schema dynamically from first row
        schema = self._infer_bq_schema(categorized_reqs[0])

        # Ensure dataset exists
        try:
            client.get_dataset(dataset_id)
        except Exception:
            print(f"⚠️ Dataset {dataset_id} not found. Creating...")
            dataset = bigquery.Dataset(f"{self.project_id}.{dataset_id}")
            dataset.location = self.location
            client.create_dataset(dataset, exists_ok=True)
            print(f"✅ Created dataset {dataset_id}")

        # Ensure table exists
        try:
            client.get_table(table_ref)
            print(f"✅ Table {table_ref} exists.")
        except Exception:
            print(f"⚠️ Table {table_ref} not found. Creating...")
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table)
            print(f"✅ Created table {table_ref}")

        # --- 🔥 Fix: stringify dicts before insert ---
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
            print(f"❌ Errors inserting rows: {errors}")
        else:
            print(f"✅ Inserted {len(rows)} categorized requirements into {table_ref}")
