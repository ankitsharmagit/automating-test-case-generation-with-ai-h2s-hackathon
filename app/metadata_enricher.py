import re
import spacy
import yaml
from tqdm import tqdm
from app.utils import get_logger   # ✅ centralized logging utility


# Initialize logger
logger = get_logger("MetadataEnricher")


class MetadataEnricher:
    """
    🏥 Metadata Enricher (Layer 3 with YAML-driven regulations)
    - Loads regulation list with details + aliases from YAML
    - Extracts regulation mentions via regex & keyword search
    - Normalizes actors, actions, data_types
    - Always maps matches back to canonical regulation names
    """

    def __init__(self, regulation_file="regulations.yaml"):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.error("❌ spaCy model missing. Run: python -m spacy download en_core_web_sm")
            raise RuntimeError("spaCy model missing. Run: python -m spacy download en_core_web_sm")

        # Load regulations from YAML
        try:
            with open(regulation_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.known_regulations = cfg.get("regulations", [])
            logger.info(f"✅ Loaded {len(self.known_regulations)} regulations from {regulation_file}")
        except Exception as e:
            logger.error(f"❌ Failed to load regulations file: {e}")
            self.known_regulations = []

        # Build alias → canonical mapping for quick lookup ✅
        self.alias_map = {}
        for reg in self.known_regulations:
            canonical = reg.get("name", "").strip()
            if not canonical:
                continue
            self.alias_map[canonical.lower()] = canonical
            for alias in reg.get("aliases", []):
                self.alias_map[alias.lower()] = canonical
        logger.info(f"✅ Built alias map with {len(self.alias_map)} entries")

        # Canonical mappings
        self.actor_map = {
            "admin": "Administrator",
            "system administrator": "Administrator",
            "compliance officer": "Compliance Officer",
            "doctor": "Clinician",
            "nurse": "Clinician",
        }

        self.data_type_map = {
            "insurance data": "Claims Data",
            "claims": "Claims Data",
            "billing information": "Claims Data",
            "patient demographics": "Patient Demographics",
            "phi": "PHI",
            "protected health information": "PHI",
        }

        self.allowed_actions = {
            "encrypt", "store", "delete", "process",
            "transmit", "access", "restrict", "audit", "validate"
        }

    # -------------------------
    # Category-specific helpers
    # -------------------------

    def _process_regulations(self, text: str):
        """Extract regulations using name, aliases, and summary keywords."""
        found = []
        logger.debug("🔍 Checking for regulations in requirement text")

        for reg in self.known_regulations:
            reg_name = reg.get("name", "")
            aliases = reg.get("aliases", [])
            reg_summary = reg.get("summary", "")

            # Direct name match
            if re.search(rf"\b{re.escape(reg_name)}\b", text, re.IGNORECASE):
                found.append(reg_name)
                logger.debug(f"   ✅ Matched regulation name: {reg_name}")
                continue

            # Alias match
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE):
                    found.append(reg_name)
                    logger.debug(f"   ✅ Matched alias '{alias}' → {reg_name}")
                    break

            # Summary keyword match
            for keyword in re.findall(r"\w+", reg_summary):
                if re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE):
                    found.append(reg_name)
                    logger.debug(f"   ✅ Matched keyword '{keyword}' → {reg_name}")
                    break
                    
        return list(set(found))  # ✅ only canonical names

    def _process_actions(self, text: str):
        """Extract verbs that match allowed compliance-related actions."""
        logger.debug("🔍 Extracting actions from requirement text")
        doc = self.nlp(text)
        verbs = [token.lemma_.lower() for token in doc if token.pos_ == "VERB"]
        filtered = [v.capitalize() for v in verbs if v in self.allowed_actions]
        if filtered:
            logger.debug(f"   ✅ Found actions: {filtered}")
        return filtered

    def _process_actors(self, actors):
        """Normalize actors to canonical names."""
        logger.debug("🔍 Normalizing actors")
        if not actors:
            return []
        normed = [self.actor_map.get(str(a).lower(), a) for a in actors]
        logger.debug(f"   ✅ Normalized actors: {normed}")
        return list(set(normed))

    def _process_data_types(self, data_types):
        """Normalize data types to canonical categories."""
        logger.debug("🔍 Normalizing data types")
        if not data_types:
            return []
        if not isinstance(data_types, list):
            data_types = [data_types]
        normed = []
        for d in data_types:
            d_low = str(d).lower()
            mapped = self.data_type_map.get(d_low, d)
            normed.append(mapped)
            if mapped != d:
                logger.debug(f"   🔄 Mapped data type '{d}' → '{mapped}'")
        return list(set(normed))

    def _save_metadata(self, req, regulation_sections):
        """Attach enrichment metadata to a requirement."""
        logger.debug(f"📝 Saving enrichment metadata for requirement {req.get('requirement_id', 'unknown')}")
        if "metadata" not in req:
            req["metadata"] = {}
        req["metadata"]["enrichment"] = {
            "domain": "healthcare",
            "regulation_sections": regulation_sections,
            "normalized": {
                "actors": req["actors"],
                "actions": req["action"],
                "data_type": req["data_type"]
            }
        }
        return req

    # -------------------------
    # Main enrichment pipeline
    # -------------------------

    def enrich(self, structured_reqs):
        enriched = []
        logger.info(f"🚀 Starting enrichment for {len(structured_reqs)} structured requirements")

        for req in tqdm(structured_reqs, desc="Enriching requirements"):
            text = req.get("statement", "")
            logger.debug(f"⚙️ Processing requirement {req.get('requirement_id', 'unknown')}")

            # Run category-specific processors
            regulation_sections = self._process_regulations(text)
            
            extra_actions = self._process_actions(text)
            # Ensure actions are always a list before merging


            req["actors"] = self._process_actors(req.get("actors", []))
            req["data_type"] = self._process_data_types(req.get("data_type", []))
            
            existing_actions = req.get("action", [])
            if isinstance(existing_actions, str):
                existing_actions = [existing_actions]
            elif not isinstance(existing_actions, list):
                existing_actions = []

            req["action"] = list(set(existing_actions + extra_actions))


            # Normalize all regulations to canonical names
            req["regulation"] = list(set(
                [self.alias_map.get(r.lower(), r) for r in req.get("regulation", [])] +
                regulation_sections
            ))
            print(req["regulation"])

            # Save metadata
            req = self._save_metadata(req, regulation_sections)

            logger.debug(f"✅ Enriched requirement {req.get('requirement_id', 'unknown')} → {regulation_sections}")
            enriched.append(req)

        logger.info(f"🎉 Completed enrichment → {len(enriched)} requirements enriched")
        return enriched
