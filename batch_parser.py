import os
import pdfplumber
import docx
import xmltodict
from bs4 import BeautifulSoup
import json

SUPPORTED_EXT = [".pdf", ".docx", ".xml", ".html", ".htm", ".json"]

class BatchParser:
    def __init__(self, data_folder="data"):
        self.data_folder = data_folder

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

    def normalize_requirements(self, raw_text, max_chunk_size=500):
        """Break raw text into requirement chunks (~500 chars)."""
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

        chunks, current = [], ""
        for line in lines:
            # Skip headings (1-3 words only, no period)
            if len(line.split()) <= 3 and "." not in line:
                continue  

            if len(current) + len(line) < max_chunk_size:
                current += " " + line
            else:
                chunks.append(current.strip())
                current = line
        if current:
            chunks.append(current.strip())

        return chunks



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

        return self.normalize_requirements(raw_text)

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
                results[file] = [f"Error parsing {file}: {e}"]
        return results
