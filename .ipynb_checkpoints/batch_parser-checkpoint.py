import os
import pdfplumber
import docx
import xmltodict
from bs4 import BeautifulSoup
import json
import uuid
import datetime
import re
import nltk
from nltk.tokenize import sent_tokenize
from google.cloud import bigquery
import string
# Load spaCy model once
# nlp = spacy.load("en_core_web_sm")

SUPPORTED_EXT = [".pdf", ".docx", ".xml", ".html", ".htm", ".json"]

def normalize_text_for_dedup(text: str) -> str:
        """Normalize requirement text for deduplication check."""
        t = text.lower().strip()
        t = re.sub(r"\s+", " ", t)  # collapse whitespace
        t = t.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
        return t
    
class BatchParser:
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder

    # ----------------- PARSERS -----------------
    def parse_pdf(self, file_path):
        text = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text())
        return "\n".join(filter(None, text))

    def parse_docx(self, file_path):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    def parse_xml(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = xmltodict.parse(f.read())
        return json.dumps(data, indent=2)

    def parse_html(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        return soup.get_text(separator="\n", strip=True)

    def parse_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
   

    
    # ----------------- REQUIREMENT EXTRACTION -----------------
    def extract_requirements(self, raw_text, filename=None):
        """
        Extract clean requirement statements from raw text.
        """
        # 1. Clean headers/footers/page numbers
        lines = raw_text.split("\n")
        clean_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r"^page\s*\d+", line.lower()):
                continue
            if re.match(r"^\d+$", line.strip()):  # only numbers
                continue
            clean_lines.append(line)

        text = " ".join(clean_lines)

        # 2. Sentence segmentation
        # doc = nlp(text)
        # Sentence segmentation (regex fallback, no external data needed)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        requirements = []
        seen = set()
        buffer = ""

        for sent in sentences:
            sent_lower = sent.lower()

            # Filter: too short
            if len(sent) < 30:
                continue

            # Filter: boilerplate/junk
            if any(k in sent_lower for k in ["copyright", "license", "creative commons",
                                             "foundation", "trademark", "confidential"]):
                continue

            # Requirement-like keywords
            req_keywords = ["shall", "must", "should", "will", "require",
                            "system", "user", "data", "information", "hipaa", "phi"]
            
            if self.is_clean_requirement(sent):
                norm = normalize_text_for_dedup(sent)
                if norm not in seen:
                    seen.add(norm)
                    requirements.append(sent)
                # Merge consecutive requirement-like sentences
                if buffer:
                    buffer += " " + sent
                    requirements.append(buffer.strip())
                    buffer = ""
                else:
                    buffer = sent
            else:
                if buffer:
                    requirements.append(buffer.strip())
                    buffer = ""

        if buffer:  # flush last
            requirements.append(buffer.strip())

        # 3. Normalize IDs
        normalized_reqs = [
            {
                "id": str(uuid.uuid4()),
                "requirement_id": f"REQ-{i+1:03d}",
                "filename": filename,
                "statement": req,
                "created_at": datetime.datetime.utcnow().isoformat()
            }
            for i, req in enumerate(requirements)
        ]

        return normalized_reqs

    # ----------------- PIPELINE -----------------
    def parse_file(self, file_path):
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".pdf":
            raw_text = self.parse_pdf(file_path)
        elif ext == ".docx":
            raw_text = self.parse_docx(file_path)
        elif ext == ".xml":
            raw_text = self.parse_xml(file_path)
        elif ext in [".html", ".htm"]:
            raw_text = self.parse_html(file_path)
        elif ext == ".json":
            raw_text = self.parse_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return self.extract_requirements(raw_text, filename=os.path.basename(file_path))

    def collect_files(self):
        """Recursively collect supported files from data folder."""
        all_files = []
        for root, _, files in os.walk(self.data_folder):
            for file in files:
                if os.path.splitext(file)[-1].lower() in SUPPORTED_EXT:
                    all_files.append(os.path.join(root, file))
        return all_files

    def parse_batch(self):
        """Parse all supported files inside the data folder."""
        files = self.collect_files()
        results = {}
        for file in files:
            try:
                results[file] = self.parse_file(file)
            except Exception as e:
                results[file] = [{"error": f"Error parsing {file}: {e}"}]
        return results

    def export_results(self, results, project_id=None, dataset_id=None,
                   table_id="requirements", save_local=True):
        """
        Export parsed requirements.
        - If BigQuery project_id & dataset_id are provided → export to BigQuery.
        - Always fallback to save locally as requirements.json if save_local=True.
        """
        rows = []
        for fname, reqs in results.items():
            for r in reqs:
                # Only keep schema fields (avoid errors in BQ)
                row = {
                    "id": r.get("id", str(uuid.uuid4())),
                    "requirement_id": r.get("requirement_id", ""),
                    "filename": r.get("filename", fname),
                    "statement": r.get("statement", ""),
                    "created_at": str(r.get("created_at", datetime.datetime.utcnow().isoformat()))
                }
                rows.append(row)

        if not rows:
            print("⚠️ No requirements found. Nothing to export.")
            return

        # --- Try BigQuery ---
        bq_success = False
        if project_id and dataset_id:
            try:
                client = bigquery.Client(project=project_id)
                table_ref = f"{project_id}.{dataset_id}.{table_id}"

                schema = [
                    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("requirement_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("filename", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("statement", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
                ]

                try:
                    client.get_table(table_ref)
                    print(f"✅ Table {table_ref} already exists.")
                except Exception:
                    print(f"⚠️ Table {table_ref} not found. Creating...")
                    table = bigquery.Table(table_ref, schema=schema)
                    client.create_table(table)
                    print(f"✅ Created table {table_ref}")

                errors = client.insert_rows_json(table_ref, rows)
                if errors:
                    print(f"❌ Errors inserting rows: {errors}")
                else:
                    print(f"✅ Inserted {len(rows)} rows into {table_ref}")
                    bq_success = True

            except Exception as e:
                print(f"⚠️ BigQuery export failed: {e}")

        # --- Always save locally if enabled ---
        if save_local:
            try:
                file_name = "requirements.json"
                with open(file_name, "w", encoding="utf-8") as f:
                    json.dump(rows, f, indent=2)
                print(f"💾 Saved {len(rows)} requirements locally → {file_name}")
            except Exception as e:
                print(f"❌ Failed to save local JSON: {e}")

        if not bq_success and not save_local:
            print("⚠️ Nothing exported (both BQ and local disabled).")
            
    def is_clean_requirement(self, text: str) -> bool:
        """Strict filter for requirement sentences."""
        t = text.strip()

        # --- Remove artifacts ---
        t = re.sub(r"\.{3,}", " ", t)          # dot leaders
        t = re.sub(r"\s*\d+\s*", " ", t)       # stray numbers
        t = re.sub(r"\s{2,}", " ", t)          # extra spaces
        t = t.strip()

        if not t or len(t) < 30 or len(t) > 350:
            return False

        # --- Heading detection (short + no punctuation) ---
        if len(t.split()) <= 6 and "." not in t:
            return False

        # --- Boilerplate ---
        junk_keywords = [
            "acknowledgment", "preface", "contributors", "support",
            "copyright", "license", "foundation", "trademark",
            "methodology", "process framework", "catalog", "diagram"
        ]
        if any(k in t.lower() for k in junk_keywords):
            return False

        # --- Must have verb (requirement/action indicator) ---
        if not re.search(r"\b(is|are|has|have|shall|must|should|require|ensure|will)\b", t.lower()):
            return False

        # --- Requirement / domain keywords ---
        req_keywords = ["shall", "must", "should", "require", "ensure", "will",
                        "system", "user", "data", "information system", "hipaa", "phi"]
        if not any(k in t.lower() for k in req_keywords):
            return False

        return True



