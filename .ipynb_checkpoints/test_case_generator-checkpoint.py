from langchain_google_vertexai import VertexAI
import uuid, json, re

class TestCaseGenerator:
    def __init__(self, model="mistralai/mistral-large-2411@001"):
        self.llm = VertexAI(model_name=model, temperature=0.2)

        self.prompt_template = """
        You are a healthcare QA expert.
        
        Requirement: "{requirement}"
        Metadata: {metadata}

        Generate 3 different test cases for this requirement:
        - One positive scenario (happy path).
        - One negative scenario (failure or invalid input).
        - One edge case (boundary condition).

        ⚠️ Output ONLY valid JSON.
        Do not include explanations, markdown, or text outside the JSON.
        The JSON must be a list of objects, where each object has:
        - test_id (UUID)
        - description (string)
        - steps (list of strings)
        - expected_result (list of strings)
        - linked_traceability_id (string)
        """

    def _extract_json(self, text):
        """Try to extract JSON substring from text using regex."""
        try:
            match = re.search(r"(\[.*\])", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except Exception:
            pass
        return None

    def generate_test_cases(self, requirement, metadata):
        prompt = self.prompt_template.format(
            requirement=requirement,
            metadata=metadata
        )
        try:
            response = self.llm.invoke(prompt)
            raw_text = response if isinstance(response, str) else getattr(response, "content", str(response))

            # Try direct parse
            try:
                result = json.loads(raw_text)
            except:
                result = self._extract_json(raw_text)

            if not result:
                raise ValueError("No valid JSON found in model output.")

            # Normalize fields
            for tc in result:
                steps = tc.get("steps", [])
                if isinstance(steps, str):
                    tc["steps"] = [s.strip() for s in steps.split("\n") if s.strip()]

                exp = tc.get("expected_result", [])
                if isinstance(exp, str):
                    tc["expected_result"] = [s.strip() for s in exp.split("\n") if s.strip()]

                if not tc.get("test_id"):
                    tc["test_id"] = str(uuid.uuid4())

                tc["linked_traceability_id"] = metadata.get("traceability_id")

            return result

        except Exception as e:
            print("⚠️ Parsing failed:", e)
            return [{
                "test_id": str(uuid.uuid4()),
                "description": "Failed to parse",
                "steps": [],
                "expected_result": [],
                "linked_traceability_id": metadata.get("traceability_id")
            }]
