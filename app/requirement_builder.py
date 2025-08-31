import asyncio
import json
import datetime
import re
import json5
from google.cloud import bigquery
from langchain_google_vertexai import VertexAI
from tqdm import tqdm


async def safe_batch_async(llm, prompts, timeout=90):
    """Run multiple LLM calls concurrently with timeout handling."""
    tasks = [llm.ainvoke(prompt) for prompt in prompts]
    try:
        return await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        print("Timeout reached, skipping this batch.")
        return ["{}" for _ in prompts]


class RequirementBuilder:
    """Requirement Builder: Converts raw requirement candidates into structured, traceable requirements."""

    def __init__(self, model="gemini-2.5-pro", project_id=None, location="us-central1"):
        self.llm = VertexAI(model_name=model, temperature=0, project=project_id, location=location)
        self.project_id=project_id

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
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"```(json)?", "", text)
            text = text.replace("```", "")
        return text.strip()

    def _normalize_for_dedup(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    def _normalize_fields(self, req: dict) -> dict:
        """Ensure regulation and acceptance_criteria are consistent lists of strings."""
        norm_reg = []
        for r in req.get("regulation", []):
            if isinstance(r, str):
                norm_reg.append(r)
            elif isinstance(r, dict):
                val = r.get("standard", "")
                if "section" in r:
                    val += f" - {r['section']}"
                if val:
                    norm_reg.append(val)
        req["regulation"] = norm_reg

        norm_ac = []
        for ac in req.get("acceptance_criteria", []):
            if isinstance(ac, str):
                norm_ac.append(ac)
            elif isinstance(ac, dict):
                norm_ac.append(ac.get("description", ""))
        req["acceptance_criteria"] = [x for x in norm_ac if x]

        return req

    def _validate_requirement(self, req, req_id, source_file=None, raw_text=None):
        return {
            "requirement_id": req.get("requirement_id", req_id),
            "category": req.get("category", "Functional"),
            "title": req.get("title", "Untitled Requirement"),
            "statement": req.get("statement", ""),
            "priority": req.get("priority", "P3"),
            "severity": req.get("severity", "Minor"),
            "regulation": req.get("regulation", []),
            "actors": req.get("actors", []),
            "data_type": req.get("data_type", []),
            "action": req.get("action", []),
            "acceptance_criteria": req.get("acceptance_criteria", []),
            "dependencies": req.get("dependencies", []),
            "traceability": req.get("traceability", []),
            "metadata": {
                "source_file": source_file,
                "llm_model": self.llm.model_name,
                "validated": True,
                "raw_input": raw_text
            },
            "created_at": datetime.datetime.utcnow().isoformat()
        }

    def _sanitize_for_bq(self, req: dict) -> dict:
        """Fix fields so they match BigQuery schema."""
        # category as string
        if isinstance(req.get("category"), list):
            req["category"] = req["category"][0] if req["category"] else "Functional"

        # data_type as repeated string
        if isinstance(req.get("data_type"), str):
            req["data_type"] = [req["data_type"]] if req["data_type"].strip() else []
        elif not req.get("data_type"):
            req["data_type"] = []
        elif isinstance(req.get("data_type"), list):
            req["data_type"] = [str(x) for x in req["data_type"] if x]

        # action as repeated string
        action_val = req.get("action")
        if isinstance(action_val, str):
            req["action"] = [action_val.strip()] if action_val.strip() else []
        elif isinstance(action_val, list):
            req["action"] = [str(x).strip() for x in action_val if x and str(x).strip()]
        else:
            req["action"] = []

        return req

    async def build_registry(self, requirements, batch_size=10):
        """Process requirements asynchronously in batches."""
        structured = []
        prompts = []
        seen = set()

        for i, req in enumerate(requirements, start=1):
            if isinstance(req, dict):
                req_id = req.get("requirement_id", f"REQ-{i:03d}")
                text = req.get("statement", "")
                source_file = req.get("filename")
            else:
                req_id = f"REQ-{i:03d}"
                text = str(req)
                source_file = None
            prompts.append((req_id, self._make_prompt(text, req_id), text, source_file))

        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing requirements"):
            batch = prompts[i:i+batch_size]
            req_ids = [r[0] for r in batch]
            req_prompts = [r[1] for r in batch]
            raw_texts = [r[2] for r in batch]
            source_files = [r[3] for r in batch]

            responses = await safe_batch_async(self.llm, req_prompts)

            for req_id, resp, raw_inp, source_file in zip(req_ids, responses, raw_texts, source_files):
                raw_text = str(resp) if isinstance(resp, str) else getattr(resp, "content", str(resp))
                raw_text = self._clean_json(raw_text)

                try:
                    parsed = json.loads(raw_text)
                except Exception:
                    try:
                        parsed = json5.loads(raw_text)
                    except Exception:
                        print(f"Failed to parse LLM output for {req_id}: {raw_text[:200]}...")
                        parsed = {}

                validated = self._validate_requirement(parsed, req_id, source_file=source_file, raw_text=raw_inp)
                validated = self._normalize_fields(validated)

                if validated["statement"]:
                    norm = self._normalize_for_dedup(validated["statement"])
                    if norm not in seen:
                        seen.add(norm)
                        structured.append(validated)
                    else:
                        print(f"Skipping duplicate requirement: {req_id}")

        return structured

    def export_to_bq(self, structured_reqs, dataset_id, table_id="requirements", batch_size=100):
        """Export structured requirements into BigQuery with safe batch inserts."""
        client = bigquery.Client(project=self.project_id)
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"

        schema = [
            bigquery.SchemaField("requirement_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("category", "STRING"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("statement", "STRING"),
            bigquery.SchemaField("priority", "STRING"),
            bigquery.SchemaField("severity", "STRING"),
            bigquery.SchemaField("regulation", "STRING", mode="REPEATED"),
            bigquery.SchemaField("actors", "STRING", mode="REPEATED"),
            bigquery.SchemaField("data_type", "STRING", mode="REPEATED"),
            bigquery.SchemaField("action", "STRING", mode="REPEATED"),
            bigquery.SchemaField("acceptance_criteria", "STRING", mode="REPEATED"),
            bigquery.SchemaField("dependencies", "STRING", mode="REPEATED"),
            bigquery.SchemaField("traceability", "STRING", mode="REPEATED"),
            bigquery.SchemaField("metadata", "JSON"),
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

        # Prepare sanitized rows
        all_rows = []
        for req in structured_reqs:
            cleaned = self._sanitize_for_bq(req)
            all_rows.append({
                **cleaned,
                "metadata": json.dumps(cleaned.get("metadata", {})),
                "created_at": datetime.datetime.utcnow().isoformat()
            })

        # Insert in batches
        total_inserted = 0
        for i in range(0, len(all_rows), batch_size):
            batch = all_rows[i:i + batch_size]
            errors = client.insert_rows_json(table_ref, batch)
            if errors:
                print(f"Errors inserting batch {i//batch_size + 1}: {errors}")
            else:
                total_inserted += len(batch)
                print(f"Inserted batch {i//batch_size + 1} ({len(batch)} rows).")

        print(f"Finished inserting {total_inserted} rows into {table_ref}")
