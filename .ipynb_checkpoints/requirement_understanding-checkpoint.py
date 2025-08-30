from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import uuid

class RequirementUnderstanding:
    def __init__(self, model="mistralai/mistral-large-2411@001"):
        self.llm = VertexAI(model_name=model, temperature=0)

        # Define schema
        response_schemas = [
            ResponseSchema(name="type", description="Functional / Non-Functional / Regulatory"),
            ResponseSchema(name="priority", description="High / Medium / Low"),
            ResponseSchema(name="compliance_tags", description="List of compliance standards"),
            ResponseSchema(name="traceability_id", description="UUID string"),
        ]

        self.parser = StructuredOutputParser.from_response_schemas(response_schemas)
        self.format_instructions = self.parser.get_format_instructions()

        self.prompt = PromptTemplate(
            template="""
            You are a healthcare QA expert.
            
            Requirement: "{requirement}"

            Carefully analyze it and output ONLY valid JSON with this structure:
            {format_instructions}

            Rules:
            - Do not add explanations, text, or commentary.
            - Always include "type", "priority", "compliance_tags", and "traceability_id".
            - For traceability_id, generate a valid UUID string.
            """,
            input_variables=["requirement"],
            partial_variables={"format_instructions": self.format_instructions},
        )

    def analyze(self, requirement_text):
        try:
            response = self.llm.invoke(self.prompt.format(requirement=requirement_text))
            
            # Extract raw text
            if isinstance(response, str):
                raw_text = response
            elif hasattr(response, "content"):
                raw_text = response.content
            else:
                raw_text = str(response)

            result = self.parser.parse(raw_text)

            # 🛠 Normalize compliance_tags
            tags = result.get("compliance_tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
                result["compliance_tags"] = tags

            return result

        except Exception as e:
            print("⚠️ Parsing failed, reason:", e)
            return {
                "type": "Unknown",
                "priority": "Unknown",
                "compliance_tags": [],
                "traceability_id": str(uuid.uuid4())
            }

