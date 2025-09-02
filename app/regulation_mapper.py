# app/regulation_mapper.py
import re
import json
import yaml
from tqdm import tqdm
from app.utils import get_logger
from app.llm_utils import safe_llm_batch_async

logger = get_logger("RegulationMapper", log_file="logs/regulation_mapper.log")


class RegulationMapper:
    """
    Maps requirements to regulations & obligations.
    - Regulations are grounded in YAML to avoid hallucinations
    - Obligations are extracted by LLM as actions/duties
    """

    def __init__(self, regulation_file="regulations.yaml", llm=None):
        self.llm = llm
        try:
            with open(regulation_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.known_regulations = cfg.get("regulations", [])
            logger.info(f"✅ Loaded {len(self.known_regulations)} regulations from {regulation_file}")
        except Exception as e:
            logger.error(f"❌ Failed to load regulations file {regulation_file}: {e}")
            self.known_regulations = []

        # Canonical list of regulation names
        self.reg_names = [reg.get("name", "").strip() for reg in self.known_regulations if reg.get("name")]

        # Build alias → canonical name mapping
        self.alias_map = {}
        for reg in self.known_regulations:
            canonical = reg.get("name", "").strip()
            if not canonical:
                continue
            self.alias_map[canonical.lower()] = canonical
            for alias in reg.get("aliases", []):
                self.alias_map[alias.lower()] = canonical

        logger.info(f"✅ Built alias map with {len(self.alias_map)} entries")

    # ------------------ Helpers ------------------

    def _parse_llm_json(self, raw_text: str):
        """Robust JSON extraction from LLM output."""
        if not raw_text:
            return {}

        text = raw_text.strip()
        text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
        text = text.replace("```", "").strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return {}

    def normalize_regulations(self, regs):
        """Normalize list of regulation strings to canonical names (based on alias map)."""
        normalized = []
        for r in regs:
            if not r:
                continue
            canonical = self.alias_map.get(str(r).lower(), None)
            if canonical:
                normalized.append(canonical)
            elif r in self.reg_names:
                normalized.append(r)
        return list(set(normalized)) if normalized else ["NA"]

    # ------------------ Main Mapper ------------------

    async def map_batch(self, texts: list[str], batch_size: int = 10):
        """
        Identify applicable regulations & obligations for a batch of requirements.
        Returns list of dicts: {"regulation": [...], "obligations": [...]}
        """
        if not self.llm:
            logger.warning("⚠️ No LLM provided. Falling back to NA.")
            return [{"regulation": ["NA"], "obligations": []} for _ in texts]

        prompts = []
        for text in texts:
            prompt = f"""
            You are a compliance classification engine.
            Given this requirement:

            "{text}"

            Tasks:
            1. Map to ONLY these regulations: {self.reg_names}
            2. Extract obligations (the specific duties/actions implied, e.g., encrypt, log, restrict access).

            Rules:
            - "regulation" must be one or more from the list, or ["NA"]
            - "obligations" is a list of short action phrases
            - Return ONLY valid JSON in this format:
              {{
                "regulation": ["HIPAA","ISO 27001"],
                "obligations": ["Encrypt PHI", "Maintain audit logs"]
              }}
            """
            prompts.append(prompt)

        all_results = []

        for i in tqdm(range(0, len(prompts), batch_size), desc="🔍 Mapping regulations & obligations"):
            batch_prompts = prompts[i:i+batch_size]
            logger.info(f"📦 Processing batch {i//batch_size+1} ({len(batch_prompts)} requirements)")

            try:
                responses = await safe_llm_batch_async(self.llm, batch_prompts)
            except Exception as e:
                logger.error(f"❌ LLM batch failed: {e}")
                all_results.extend([{"regulation": ["NA"], "obligations": []}] * len(batch_prompts))
                continue

            for raw_text in responses:
                raw_text = str(raw_text).strip()
                parsed = self._parse_llm_json(raw_text)

                regs = self.normalize_regulations(parsed.get("regulation", []))
                obligations = parsed.get("obligations", [])
                if not isinstance(obligations, list):
                    obligations = [str(obligations)]

                all_results.append({
                    "regulation": regs,
                    "obligations": obligations
                })

        logger.info(f"✅ Completed mapping → {len(all_results)} requirements processed")
        return all_results
