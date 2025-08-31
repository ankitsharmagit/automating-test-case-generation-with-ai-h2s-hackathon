from google.cloud import bigquery
import re
import datetime

class CoverageValidator:
    def __init__(self, project_id, dataset_id="requirements_dataset"):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id

    def validate_ids(self, req_id, test_ids):
        """Regex validation for requirement + test IDs"""
        req_ok = bool(re.match(r"^REQ-\d+$", req_id))
        test_ok = all(re.match(r"^TC_[A-Z]+_\d+$", tid) for tid in test_ids)
        return req_ok and test_ok

    def build_traceability_matrix(
        self, 
        requirements_table="requirements", 
        testcases_table="test_cases", 
        output_table="traceability_matrix"
    ):
        """Generate traceability matrix with coverage validation + gap analysis"""

        query = f"""
        SELECT 
            r.requirement_id,
            ARRAY_AGG(DISTINCT t.test_id IGNORE NULLS) AS test_case_ids,
            ARRAY_AGG(DISTINCT t.type IGNORE NULLS) AS test_case_types,
            ARRAY_AGG(DISTINCT t.title IGNORE NULLS) AS test_case_titles,
            ARRAY_AGG(DISTINCT reg) AS compliance
        FROM `{self.project_id}.{self.dataset_id}.{requirements_table}` r
        LEFT JOIN `{self.project_id}.{self.dataset_id}.{testcases_table}` t
            ON r.requirement_id = t.requirement_id
        LEFT JOIN UNNEST(r.regulation) AS reg
        GROUP BY r.requirement_id
        """

        results = [dict(row) for row in self.client.query(query).result()]

        matrix = []
        for row in results:
            req_id = row["requirement_id"]
            test_case_ids = row.get("test_case_ids", []) or []
            test_case_types = [t.lower() for t in (row.get("test_case_types") or [])]
            test_case_titles = row.get("test_case_titles", []) or []

            # 🔹 Infer type from titles
            for title in test_case_titles:
                title_lower = title.lower()
                if "positive" in title_lower:
                    test_case_types.append("positive")
                elif "negative" in title_lower:
                    test_case_types.append("negative")
                elif "edge" in title_lower:
                    test_case_types.append("edge")

            # Coverage checks
            has_pos = any("positive" in t for t in test_case_types)
            has_neg = any("negative" in t for t in test_case_types)
            has_edge = any("edge" in t for t in test_case_types)

            total_types = 3  # positive, negative, edge
            covered = sum([has_pos, has_neg, has_edge])
            coverage_percent = int((covered / total_types) * 100)

            if covered == 3:
                coverage_status = "FULL"
                coverage_gaps = []
            elif covered > 0:
                coverage_status = "PARTIAL"
                coverage_gaps = []
                if not has_pos:
                    coverage_gaps.append("missing_positive")
                if not has_neg:
                    coverage_gaps.append("missing_negative")
                if not has_edge:
                    coverage_gaps.append("missing_edge")
            else:
                coverage_status = "NONE"
                coverage_gaps = ["missing_positive", "missing_negative", "missing_edge"]

            valid_ids = self.validate_ids(req_id, test_case_ids)

            matrix.append({
                "requirement_id": req_id,
                "use_case_id": None,  # placeholder
                "test_case_ids": test_case_ids,
                "coverage_status": coverage_status,
                "coverage_percent": coverage_percent,
                "coverage_gaps": coverage_gaps,
                "compliance": row.get("compliance", []),
                "created_at": datetime.datetime.utcnow().isoformat(),
                "valid_ids": valid_ids
            })

        # Save to BigQuery
        table_ref = f"{self.project_id}.{self.dataset_id}.{output_table}"

        schema = [
            bigquery.SchemaField("requirement_id", "STRING"),
            bigquery.SchemaField("use_case_id", "STRING"),
            bigquery.SchemaField("test_case_ids", "STRING", mode="REPEATED"),
            bigquery.SchemaField("coverage_status", "STRING"),
            bigquery.SchemaField("coverage_percent", "INTEGER"),
            bigquery.SchemaField("coverage_gaps", "STRING", mode="REPEATED"),
            bigquery.SchemaField("compliance", "STRING", mode="REPEATED"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
            bigquery.SchemaField("valid_ids", "BOOL"),
        ]

        # Ensure table exists (drop & recreate if schema mismatch)
        try:
            table = self.client.get_table(table_ref)
            existing_fields = {f.name for f in table.schema}
            expected_fields = {f.name for f in schema}
            if existing_fields != expected_fields:
                print(f"⚠️ Schema mismatch detected. Recreating table {table_ref}")
                self.client.delete_table(table_ref, not_found_ok=True)
                table = bigquery.Table(table_ref, schema=schema)
                self.client.create_table(table)
                print(f"✅ Recreated table {table_ref} with new schema")
        except Exception:
            table = bigquery.Table(table_ref, schema=schema)
            self.client.create_table(table)
            print(f"✅ Created table {table_ref} with schema")

        # Now insert rows
        errors = self.client.insert_rows_json(table_ref, matrix)
        if errors:
            print(f"❌ Errors inserting rows: {errors}")
        else:
            print(f"✅ Inserted {len(matrix)} rows into {table_ref}")

        return matrix
