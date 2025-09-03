# app/regulation_mapper.py
import re
import json
import yaml
from tqdm import tqdm
from app.utils import get_logger, Utils

logger = get_logger("RegulationMapper")


class RegulationMapper:
    """
    Maps requirements to regulations & obligations.
    - Strictly grounded in YAML regulation list
    - LLM primary engine, regex/keyword fallback
    """

    def __init__(self, regulation_file="regulations.yaml", model=None, project_id=None, location="us-central1"):
        from langchain_google_vertexai import VertexAI

        self.llm = VertexAI(
            model_name=model,
            temperature=0,
            project=project_id,
            location=location
        )
        self.utils = Utils()

        try:
            with open(regulation_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.known_regulations = cfg.get("regulations", [])
            logger.info(f"Loaded {len(self.known_regulations)} regulations from {regulation_file}")
        except Exception as e:
            logger.error(f"Failed to load regulations file {regulation_file}: {e}")
            self.known_regulations = []

        # Canonical list of regulation names
        self.reg_names = [reg.get("name", "").strip() for reg in self.known_regulations if reg.get("name")]

        # Build alias → canonical name mapping
        self.alias_map = {}
        self.regex_map = {}
        for reg in self.known_regulations:
            canonical = reg.get("name", "").strip()
            if not canonical:
                continue
            self.alias_map[canonical.lower()] = canonical
            for alias in reg.get("aliases", []):
                self.alias_map[alias.lower()] = canonical

            # Add regex/keywords from YAML if present
            if "keywords" in reg:
                self.regex_map[canonical] = [re.compile(k, re.IGNORECASE) for k in reg["keywords"]]

        logger.info(f"Built alias map with {len(self.alias_map)} entries")
        logger.info(f"Loaded regex keyword map for {len(self.regex_map)} regulations")

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
        """Normalize regulation strings to canonical YAML names."""
        normalized = []
        for r in regs:
            if not r:
                continue
            r_clean = str(r).lower().strip()
            # Exact or alias match
            canonical = self.alias_map.get(r_clean)
            if canonical:
                normalized.append(canonical)
                continue
            # Partial/fuzzy match
            for key, canonical_name in self.alias_map.items():
                if key in r_clean:
                    normalized.append(canonical_name)
                    break
        return list(set(normalized)) if normalized else ["NA"]

    def regex_fallback(self, text: str):
        """Deterministic regex-based mapping from YAML keywords."""
        matches = []
        for reg, patterns in self.regex_map.items():
            if any(p.search(text) for p in patterns):
                matches.append(reg)
        return matches if matches else ["NA"]

    # ------------------ Main Mapper ------------------

    async def map_batch(self, texts: list[str], batch_size: int = 10):
        """
        Identify applicable regulations & obligations for a batch of requirements.
        Returns list of dicts: {"regulation": [...], "obligations": [...]}
        """
        if not self.llm:
            logger.warning("No LLM provided. Falling back to NA.")
            return [{"regulation": self.regex_fallback(t), "obligations": []} for t in texts]

        prompts = []
        reg_list_str = "\n".join([f"- {r}" for r in self.reg_names])
        for text in texts:
            prompt = f"""
            You are a compliance classification engine.

            Requirement:
            "{text}"

            Task:
            1. Choose ONLY from this regulation list:
            {reg_list_str}

            2. Extract obligations (specific duties/actions implied, e.g., encrypt, log, restrict access).

            Rules:
            - "regulation" must be selected strictly from the above list.
            - If no regulation clearly applies, return ["NA"].
            - "obligations" is a list of short action phrases.
            - Output ONLY valid JSON in this format:
              {{
                "regulation": ["HIPAA"],
                "obligations": ["Encrypt PHI", "Maintain audit logs"]
              }}
            """
            prompts.append(prompt)

        all_results = []

        for i in tqdm(range(0, len(prompts), batch_size), desc="Mapping regulations & obligations"):
            batch_prompts = prompts[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} ({len(batch_prompts)} requirements)")

            try:
                responses = await self.utils.safe_llm_batch(
                    self.llm,
                    batch_prompts,
                    batch_size=len(batch_prompts)
                )
            except Exception as e:
                logger.error(f"LLM batch failed: {e}")
                # fallback all with regex
                for t in batch_prompts:
                    all_results.append({
                        "regulation": self.regex_fallback(t),
                        "obligations": []
                    })
                continue

            for raw_text, original_text in zip(responses, batch_prompts):
                raw_text = str(raw_text).strip()
                parsed = self._parse_llm_json(raw_text)

                regs = self.normalize_regulations(parsed.get("regulation", []))
                obligations = parsed.get("obligations", [])
                if not isinstance(obligations, list):
                    obligations = [str(obligations)]

                # Hybrid fallback: if NA → regex mapping
                if regs == ["NA"]:
                    regs = self.regex_fallback(original_text)

                logger.debug(f"Mapped regulation={regs}, obligations={obligations}")

                all_results.append({
                    "regulation": regs,
                    "obligations": obligations
                })

        logger.info(f"Completed mapping → {len(all_results)} requirements processed")
        return all_results
