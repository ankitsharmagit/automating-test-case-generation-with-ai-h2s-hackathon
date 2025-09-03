import os
import pdfplumber
import docx
import xmltodict
from bs4 import BeautifulSoup
import json
import uuid
import datetime
import re
import string
from app.utils import get_logger   # ✅ centralized logging utility

# Initialize logger
logger = get_logger("BatchParser")

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
        logger.info(f"Initialized BatchParser with data_folder={data_folder}")

    # ----------------- PARSERS -----------------
    def parse_pdf(self, file_path):
        logger.debug(f"Parsing PDF: {file_path}")
        text = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text())
        return "\n".join(filter(None, text))

    def parse_docx(self, file_path):
        logger.debug(f"Parsing DOCX: {file_path}")
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    def parse_xml(self, file_path):
        logger.debug(f"Parsing XML: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = xmltodict.parse(f.read())
        return json.dumps(data, indent=2)

    def parse_html(self, file_path):
        logger.debug(f"Parsing HTML: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        return soup.get_text(separator="\n", strip=True)

    def parse_json(self, file_path):
        logger.debug(f"Parsing JSON: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2)

    # ----------------- REQUIREMENT EXTRACTION -----------------
    def extract_requirements(self, raw_text, filename=None):
        """Extract clean requirement statements from raw text."""
        logger.debug(f"Extracting requirements from file={filename or 'unknown'}")

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

        # 2. Sentence segmentation (regex fallback)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        requirements = []
        seen = set()
        buffer = ""

        for sent in sentences:
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

        logger.info(f"Extracted {len(normalized_reqs)} requirements from {filename}")
        return normalized_reqs

    # ----------------- PIPELINE -----------------
    def parse_file(self, file_path):
        logger.info(f"Parsing file: {file_path}")
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
            logger.error(f"Unsupported file format: {ext}")
            raise ValueError(f"Unsupported file format: {ext}")

        return self.extract_requirements(raw_text, filename=os.path.basename(file_path))

    def collect_files(self):
        """Recursively collect supported files from data folder."""
        all_files = []
        for root, _, files in os.walk(self.data_folder):
            for file in files:
                if os.path.splitext(file)[-1].lower() in SUPPORTED_EXT:
                    all_files.append(os.path.join(root, file))
        logger.info(f"Collected {len(all_files)} supported files from {self.data_folder}")
        return all_files

    def parse_batch(self):
        """Parse all supported files inside the data folder."""
        files = self.collect_files()
        results = {}
        for file in files:
            try:
                results[file] = self.parse_file(file)
            except Exception as e:
                logger.error(f"Error parsing {file}: {e}")
                results[file] = [{"error": f"Error parsing {file}: {e}"}]
        logger.info(f"Finished parsing batch → {len(results)} files processed")
        return results

    # ----------------- REQUIREMENT FILTER -----------------
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

        # --- Heading detection ---
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

        # --- Must have verb ---
        if not re.search(r"\b(is|are|has|have|shall|must|should|require|ensure|will)\b", t.lower()):
            return False

        # --- Requirement / domain keywords ---
        req_keywords = ["shall", "must", "should", "require", "ensure", "will",
                        "system", "user", "data", "information system", "hipaa", "phi"]
        if not any(k in t.lower() for k in req_keywords):
            return False

        return True
