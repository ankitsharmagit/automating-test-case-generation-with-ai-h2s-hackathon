import asyncio
import json
import datetime
import re
import json5
from langchain_google_vertexai import VertexAI
from tqdm import tqdm

from app.utils import Utils, get_logger

# Initialize logger
logger = get_logger("RequirementBuilder")


class RequirementBuilder:
    """Requirement Builder: Converts raw requirement candidates into structured, traceable requirements."""

    def __init__(self, model=None, project_id=None, location="us-central1"):
        self.llm = VertexAI(
            model_name=model, temperature=0, project=project_id, location=location
        )
        self.project_id = project_id
        self.utils = Utils()
        logger.info(
            f"Initialized RequirementBuilder with project_id={project_id}, model={model}, location={location}"
        )

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
        """Remove markdown fences from JSON output."""
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

        logger.debug(f"Normalized requirement {req.get('requirement_id', 'unknown')}")
        return req

    def _validate_requirement(self, req, req_id, source_file=None, raw_text=None):
        """Validate requirement structure and apply defaults."""
        logger.debug(f"Validating requirement {req_id} from {source_file}")
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
                "raw_input": raw_text,
            },
            "created_at": datetime.datetime.utcnow().isoformat(),
        }

    async def build_registry(self, requirements, batch_size=10):
        """Process requirements asynchronously in batches."""
        structured = []
        prompts = []
        seen = set()

        logger.info(
            f"Starting build_registry with {len(requirements)} requirements (batch_size={batch_size})"
        )

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
            batch = prompts[i : i + batch_size]
            req_ids = [r[0] for r in batch]
            req_prompts = [r[1] for r in batch]
            raw_texts = [r[2] for r in batch]
            source_files = [r[3] for r in batch]

            logger.info(f"Processing batch {i // batch_size + 1} ({len(batch)} requirements)")

            # Use Utils.safe_llm_batch instead of old helper
            responses = await self.utils.safe_llm_batch(
                self.llm, req_prompts, batch_size=batch_size
            )

            for req_id, resp, raw_inp, source_file in zip(
                req_ids, responses, raw_texts, source_files
            ):
                raw_text = str(resp) if isinstance(resp, str) else getattr(resp, "content", str(resp))
                raw_text = self._clean_json(raw_text)

                try:
                    parsed = json.loads(raw_text)
                except Exception:
                    try:
                        parsed = json5.loads(raw_text)
                    except Exception:
                        logger.error(
                            f"Failed to parse LLM output for {req_id}: {raw_text[:200]}..."
                        )
                        parsed = {}

                validated = self._validate_requirement(
                    parsed, req_id, source_file=source_file, raw_text=raw_inp
                )
                validated = self._normalize_fields(validated)

                if validated["statement"]:
                    norm = self._normalize_for_dedup(validated["statement"])
                    if norm not in seen:
                        seen.add(norm)
                        structured.append(validated)
                        logger.debug(f"Added structured requirement {req_id}")
                    else:
                        logger.warning(f"Skipping duplicate requirement {req_id}")

        logger.info(f"Completed build_registry → {len(structured)} structured requirements")
        return structured
