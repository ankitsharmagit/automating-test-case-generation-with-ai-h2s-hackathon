import re
import spacy
import yaml
from tqdm import tqdm


class MetadataEnricher:
    """
    🏥 Metadata Enricher (Layer 3 with YAML-driven regulations)
    - Loads regulation list from YAML
    - Extracts regulation mentions via regex & keyword search
    - Normalizes actors, actions, data_types
    - Enriches requirements without overwriting Layer 2 fields
    """

    def __init__(self, regulation_file="regulations.yaml"):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("spaCy model missing. Run: python -m spacy download en_core_web_sm")

        # Load regulations from YAML
        with open(regulation_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.known_regulations = cfg.get("regulations", [])

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

    # --- Regulation extraction via regex + YAML list ---
    def _extract_regulations(self, text):
        found = []
        for reg in self.known_regulations:
            if re.search(rf"\b{re.escape(reg)}\b", text, re.IGNORECASE):
                found.append(reg)
        return list(set(found))

    # --- Verb (action) extraction & filtering ---
    def _extract_actions(self, text):
        doc = self.nlp(text)
        verbs = [token.lemma_.lower() for token in doc if token.pos_ == "VERB"]
        return [v.capitalize() for v in verbs if v in self.allowed_actions]

    # --- Normalization helpers ---
    def _normalize_actors(self, actors):
        return list(set([self.actor_map.get(a.lower(), a) for a in actors]))

    def _normalize_data_types(self, data_types):
        if not isinstance(data_types, list):
            data_types = [data_types]
        normed = []
        for d in data_types:
            d_low = d.lower()
            normed.append(self.data_type_map.get(d_low, d))
        return list(set(normed))

    # --- Enrichment pipeline ---
    def enrich(self, structured_reqs):
        enriched = []
        for req in tqdm(structured_reqs, desc="Enriching requirements"):
            text = req.get("statement", "")

            # Extract new findings
            regulation_sections = self._extract_regulations(text)
            extra_actions = self._extract_actions(text)

            # Merge with existing Layer 2
            req["actors"] = self._normalize_actors(req.get("actors", []))
            req["data_type"] = self._normalize_data_types(req.get("data_type", []))
            req["action"] = list(set(req.get("action", []) + extra_actions))
            req["regulation"] = list(set(req.get("regulation", []) + regulation_sections))

            # Save enrichment metadata
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
            enriched.append(req)

        return enriched
