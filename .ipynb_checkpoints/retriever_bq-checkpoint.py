import uuid
from google.cloud import bigquery
from langchain_google_vertexai import VertexAIEmbeddings
import time
import numpy as np

class RequirementRetrieverBQ:
    def __init__(self, dataset_id="test_dataset", table_id="requirements", embedding_model="text-embedding-005"):
        self.client = bigquery.Client()
        self.project_id = self.client.project
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.table_ref = f"{self.project_id}.{self.dataset_id}.{self.table_id}"

        print(f"🚀 Initializing RequirementRetrieverBQ with embedding model: {embedding_model}")
        self.embeddings = VertexAIEmbeddings(model_name=embedding_model)

        # Ensure dataset + table exist
        self._ensure_dataset_exists()
        self._ensure_table_exists()

    def _ensure_dataset_exists(self):
        dataset_ref = bigquery.Dataset(f"{self.project_id}.{self.dataset_id}")
        try:
            self.client.get_dataset(dataset_ref)
            print(f"✅ Dataset {self.dataset_id} already exists.")
        except Exception:
            print(f"⚠️ Dataset {self.dataset_id} not found. Creating it now...")
            self.client.create_dataset(dataset_ref)
            print(f"✅ Created dataset {self.dataset_id}.")

    def _ensure_table_exists(self):
        dataset_ref = bigquery.DatasetReference(self.project_id, self.dataset_id)
        table_ref = dataset_ref.table(self.table_id)

        try:
            self.client.get_table(table_ref)
            print(f"✅ Table {self.table_ref} already exists.")
        except Exception:
            print(f"⚠️ Table {self.table_ref} not found. Creating it now...")

            schema = [
                bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("filename", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("chunk_id", "INT64", mode="REQUIRED"),
                bigquery.SchemaField("requirement_text", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
                bigquery.SchemaField("metadata", "JSON", mode="NULLABLE"),
            ]

            table = bigquery.Table(table_ref, schema=schema)
            self.client.create_table(table)
            print(f"✅ Created table {self.table_ref} with schema.")

    def add_requirements(self, parsed_results, batch_size=200):
        total_chunks = sum(len(chunks) for chunks in parsed_results.values())
        print(f"📦 Preparing to embed and insert {total_chunks} requirement chunks...")

        # 🔍 Fetch existing keys from BigQuery (filename + chunk_id)
        sql = f"""
        SELECT filename, chunk_id
        FROM `{self.table_ref}`
        """
        existing = {(row.filename, row.chunk_id) for row in self.client.query(sql).result()}
        print(f"ℹ️ Found {len(existing)} existing chunks in BigQuery. Will skip those.")

        rows = []
        chunk_count, skipped = 0, 0

        for fname, chunks in parsed_results.items():
            for i, chunk in enumerate(chunks):
                if (fname, i) in existing:
                    skipped += 1
                    continue  # Skip if already in BQ

                chunk_count += 1
                embedding = self.embeddings.embed_query(chunk)
                rows.append({
                    "id": str(uuid.uuid4()),
                    "filename": fname,
                    "chunk_id": i,
                    "requirement_text": chunk,
                    "embedding": embedding,
                    "metadata": None
                })

                if len(rows) >= batch_size:
                    self._insert_batch(rows)
                    rows = []

        if rows:
            self._insert_batch(rows)

        print(f"✅ Finished. Inserted {chunk_count} new chunks, skipped {skipped} existing.")

    def cosine_similarity(vec_a, vec_b):
        """Compute cosine similarity between two vectors."""
        vec_a, vec_b = np.array(vec_a), np.array(vec_b)
        return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))

    def search(self, query, top_k=5, use_bq_vectors=False):
        print(f"🔍 Running semantic search for query: {query}")
        query_embedding = self.embeddings.embed_query(query)

        if use_bq_vectors:
            # Try native BigQuery VECTOR_DOT_PRODUCT (if supported in your region)
            sql = f"""
            SELECT
              DISTINCT
              filename,
              chunk_id,
              requirement_text,
              VECTOR_DOT_PRODUCT(embedding, @query_embedding) AS similarity
            FROM `{self.table_ref}`
            ORDER BY similarity DESC
            LIMIT {top_k}
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", query_embedding)
                ]
            )
            print("📡 Executing BigQuery vector search...")
            return self.client.query(sql, job_config=job_config).result()
        else:
            # ✅ Python-side similarity (fast for <50k rows)
            sql = f"""
            SELECT DISTINCT filename, chunk_id, requirement_text, embedding
            FROM `{self.table_ref}`
            """
            rows = list(self.client.query(sql).result())

            print(f"📡 Retrieved {len(rows)} rows from BigQuery, computing similarity in Python...")
            scored = []
            for row in rows:
                score = cosine_similarity(row.embedding, query_embedding)
                scored.append((row.filename, row.chunk_id, row.requirement_text, score))

            # Sort by score and return top_k
            scored = sorted(scored, key=lambda x: x[3], reverse=True)[:top_k]

            # Mimic BigQuery Row-like objects for consistency
            from collections import namedtuple
            ResultRow = namedtuple("ResultRow", ["filename", "chunk_id", "requirement_text", "similarity"])
            return [ResultRow(*s) for s in scored]



    
    def add_requirements(self, parsed_results, batch_size=200):
        rows = []
        total_chunks = sum(len(chunks) for chunks in parsed_results.values())
        print(f"📦 Preparing to embed and insert {total_chunks} requirement chunks...")

        chunk_count = 0
        for fname, chunks in parsed_results.items():
            print(f"📄 Processing file: {fname} with {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                chunk_count += 1
                embedding = self.embeddings.embed_query(chunk)

                rows.append({
                    "id": str(uuid.uuid4()),
                    "filename": fname,
                    "chunk_id": i,
                    "requirement_text": chunk,
                    "embedding": embedding,
                    "metadata": None
                })

                # If batch is full → insert
                if len(rows) >= batch_size:
                    self._insert_batch(rows)
                    rows = []

        # Insert any leftovers
        if rows:
            self._insert_batch(rows)

        print(f"✅ Finished inserting {chunk_count} requirement chunks into BigQuery.")

    def _insert_batch(self, rows):
        print(f"📝 Inserting batch of {len(rows)} rows into {self.table_ref}...")
        errors = self.client.insert_rows_json(self.table_ref, rows)

        if errors:
            print(f"❌ Errors inserting batch: {errors}")
        else:
            print(f"✅ Successfully inserted {len(rows)} rows.")

