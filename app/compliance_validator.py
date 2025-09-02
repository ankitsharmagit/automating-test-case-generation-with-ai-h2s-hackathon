import re
import yaml
from collections import defaultdict
from app.utils import get_logger

logger = get_logger("ComplianceValidator", log_file="logs/compliance_validator.log")


class ComplianceValidator:
    def __init__(self, regulation_file="regulations.yaml"):
        # Load YAML
        with open(regulation_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.regulations = cfg.get("regulations", [])

        # Build alias map
        self.alias_map = {}
        for reg in self.regulations:
            name = reg["name"]
            self.alias_map[name.lower()] = name
            for alias in reg.get("aliases", []):
                self.alias_map[alias.lower()] = name

        logger.info(f"Loaded {len(self.regulations)} regulations")

    def _match_regulations(self, text):
        """Match text against regulation names/aliases"""
        found = []
        for alias, canonical in self.alias_map.items():
            if re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE):
                found.append(canonical)
        return list(set(found))

    def _match_obligations(self, text, regulation):
        """Match obligations from regulation summary"""
        matched = []
        for obligation in regulation.get("obligations", []):
            if re.search(rf"\b{re.escape(obligation.split()[0])}\b", text, re.IGNORECASE):
                matched.append(obligation)
        return matched

    def validate(self, requirements):
        """
        Validate requirements against compliance regulations.
        Returns:
            - enriched requirements with compliance details
            - coverage report
        """
        enriched = []
        coverage = defaultdict(lambda: {"covered": set(), "missing": set()})

        for reg in self.regulations:
            coverage[reg["name"]]["missing"] = set(reg.get("obligations", []))

        for req in requirements:
            text = req.get("statement", "")
            matched_regs = self._match_regulations(text)

            req_compliance = []
            for reg_name in matched_regs:
                reg = next(r for r in self.regulations if r["name"] == reg_name)
                matched_obligations = self._match_obligations(text, reg)

                # Update coverage
                coverage[reg_name]["covered"].update(matched_obligations)
                coverage[reg_name]["missing"].difference_update(matched_obligations)

                req_compliance.append({
                    "regulation": reg_name,
                    "obligations": matched_obligations
                })

            req["compliance"] = req_compliance
            enriched.append(req)

        # Build summary report
        summary = {}
        for reg_name, data in coverage.items():
            total = len(data["covered"]) + len(data["missing"])
            summary[reg_name] = {
                "covered": list(data["covered"]),
                "missing": list(data["missing"]),
                "coverage_pct": (len(data["covered"]) / total * 100) if total else 0
            }

        return enriched, summary
