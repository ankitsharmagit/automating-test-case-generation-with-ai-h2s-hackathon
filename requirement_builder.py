import uuid
import json
import datetime
from google.cloud import bigquery
from langchain_google_vertexai import VertexAI
from tqdm import tqdm
import concurrent.futures
import re
import json5  # pip install json5


def safe_batch(llm, prompts, timeout=90):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(llm.batch, prompts)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print("⏳ Timeout reached, skipping this batch.")
            return ["{}" for _ in prompts]


class RequirementBuilder:
    """
    🏥 Requirement Builder
    Converts raw requirement candidates into structured, traceable requirements.
    """

    def __init__(self, model="gemini-2.5-pro", project_id=None, location="us-central1"):
        self.llm = VertexAI(model_name=model, temperature=0, project=project_id, location=location)

    def _make_prompt(self, req_text, req_id):
        return f"""
        You are a healthcare QA and compliance expert.
        Convert the following requirement into a structured JSON object:

        Requirement: "{req_text}"

        Fields:
        - requirement_id: "{req_id}"
        - category: Functional / Performance / Security / Usability / Reliability / Compliance
        - title: a short name
        - statement: detailed requirement
        - priority: P1, P2, P3
        - severity: Critical, Major, Minor, Cosmetic
        - regulation: list of relevant standards (HIPAA, FDA, ISO, GDPR) with sections if any
        - actors: list of roles (clinician, patient, admin, etc.)
        - data_type: PHI, lab results, prescriptions, etc.
        - action: access, modify, delete, encrypt, etc.
        - acceptance_criteria: measurable validation points
        - dependencies: list of requirement IDs if any
        - traceability: leave as []

        Return ONLY valid JSON. No markdown, no explanation.
        """

    def _clean_json(self, text: str) -> str:
        """Remove markdown fences and clean text for JSON parsing."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"```(json)?", "", text)
            text = text.replace("```", "")
        return text.strip()

    def _normalize_for_dedup(self, text: str) -> str:
        """Lowercase & collapse whitespace for deduplication."""
        t = text.lower().strip()
        return re.sub(r"\s+", " ", t)

    def _normalize_fields(self, req: dict) -> dict:
        """Ensure regulation + acceptance_criteria are consistent lists of strings."""
        # Normalize regulation
        norm_reg = []
        for r in req.get("regulation", []):
            if isinstance(r, str):
                norm_reg.append(r)
            elif isinstance(r, dict):
                # If structured, take standard + section
                val = r.get("standard", "")
                if "section" in r:
                    val += f" - {r['section']}"
                if val:
                    norm_reg.append(val)
        req["regulation"] = norm_reg

        # Normalize acceptance_criteria
        norm_ac = []
        for ac in req.get("acceptance_criteria", []):
            if isinstance(ac, str):
                norm_ac.append(ac)
            elif isinstance(ac, dict):
                norm_ac.append(ac.get("description", ""))
        req["acceptance_criteria"] = [x for x in norm_ac if x]

        return req

    def _validate_requirement(self, req, req_id, source_file=None, raw_text=None):
        """Ensure structured requirement has all mandatory fields with defaults."""
        return {
            "requirement_id": req.get("requirement_id", req_id),
            "category": req.get("category", "Functional"),
            "title": req.get("title", "Untitled Requirement"),
            "statement": req.get("statement", ""),
            "priority": req.get("priority", "P3"),
            "severity": req.get("severity", "Minor"),
            "regulation": req.get("regulation", []),
            "actors": req.get("actors", []),
            "data_type": req.get("data_type", ""),
            "action": req.get("action", ""),
            "acceptance_criteria": req.get("acceptance_criteria", []),
            "dependencies": req.get("dependencies", []),
            "traceability": req.get("traceability", []),
            "metadata": {
                "source_file": source_file,
                "llm_model": self.llm.model_name,
                "validated": True,
                "raw_input": raw_text  # keep original for traceability
            },
            "created_at": datetime.datetime.utcnow().isoformat()
        }

    def build_registry(self, requirements, batch_size=10):
        """
        Process requirements in batches with progress tracking.
        Supports input as:
          - list of dicts (from Layer 1: {"requirement_id","statement","filename"})
          - list of strings (raw requirement text)
        """
        structured = []
        prompts = []
        seen = set()

        # Prepare prompts
        for i, req in enumerate(requirements, start=1):
            if isinstance(req, dict):  # Layer 1 dict
                req_id = req.get("requirement_id", f"REQ-{i:03d}")
                text = req.get("statement", "")
                source_file = req.get("filename")
            else:  # plain string fallback
                req_id = f"REQ-{i:03d}"
                text = str(req)
                source_file = None

            prompts.append((req_id, self._make_prompt(text, req_id), text, source_file))

        # Batch through the LLM
        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing requirements"):
            batch = prompts[i:i+batch_size]
            req_ids = [r[0] for r in batch]
            req_prompts = [r[1] for r in batch]
            raw_texts = [r[2] for r in batch]
            source_files = [r[3] for r in batch]   # ✅ FIX: added this

            responses = safe_batch(self.llm, req_prompts)

            for req_id, resp, raw_inp, source_file in zip(req_ids, responses, raw_texts, source_files):
                raw_text = resp if isinstance(resp, str) else getattr(resp, "content", str(resp))
                raw_text = self._clean_json(raw_text)

                try:
                    parsed = json.loads(raw_text)
                except Exception:
                    try:
                        parsed = json5.loads(raw_text)
                    except Exception:
                        print(f"⚠️ Failed to parse LLM output for {req_id}: {raw_text[:200]}...")
                        parsed = {}

                validated = self._validate_requirement(parsed, req_id, source_file=source_file, raw_text=raw_inp)
                validated = self._normalize_fields(validated)

                # Deduplication
                if validated["statement"]:
                    norm = self._normalize_for_dedup(validated["statement"])
                    if norm not in seen:
                        seen.add(norm)
                        structured.append(validated)
                    else:
                        print(f"⚠️ Skipping duplicate requirement: {req_id}")

        return structured


    def export_to_bq(self, structured_reqs, project_id, dataset_id, table_id="requirements"):
        """
        Export structured requirements into BigQuery.
        """
        client = bigquery.Client(project=project_id)
        table_ref = f"{project_id}.{dataset_id}.{table_id}"

        schema = [
            bigquery.SchemaField("requirement_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("category", "STRING"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("statement", "STRING"),
            bigquery.SchemaField("priority", "STRING"),
            bigquery.SchemaField("severity", "STRING"),
            bigquery.SchemaField("regulation", "STRING", mode="REPEATED"),
            bigquery.SchemaField("actors", "STRING", mode="REPEATED"),
            bigquery.SchemaField("data_type", "STRING"),
            bigquery.SchemaField("action", "STRING"),
            bigquery.SchemaField("acceptance_criteria", "STRING", mode="REPEATED"),
            bigquery.SchemaField("dependencies", "STRING", mode="REPEATED"),
            bigquery.SchemaField("traceability", "STRING", mode="REPEATED"),
            bigquery.SchemaField("metadata", "JSON"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
        ]

        # Ensure table exists
        try:
            client.get_table(table_ref)
            print(f"✅ Table {table_ref} exists.")
        except Exception:
            print(f"⚠️ Table {table_ref} not found. Creating...")
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table)
            print(f"✅ Created table {table_ref}")

        # Insert rows
        rows = []
        for req in structured_reqs:
            rows.append({
                **req,
                "metadata": json.dumps(req.get("metadata", {})),  # ensure JSON serializable
                "created_at": datetime.datetime.utcnow().isoformat()
            })

        errors = client.insert_rows_json(table_ref, rows)
        if errors:
            print(f"❌ Errors inserting rows: {errors}")
        else:
            print(f"✅ Inserted {len(rows)} structured requirements into {table_ref}")
