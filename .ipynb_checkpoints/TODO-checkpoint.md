# 📝 To-Do List for PoC → Production

---

## Phase 1 – PoC (2–3 Weeks)

**Goal:** Get a working pipeline (upload doc → AI → test cases → CSV/Excel).

### Environment Setup
- [ ] Create a **Google Cloud Project**.
- [ ] Enable **Vertex AI, Cloud Storage, Cloud Run**.
- [ ] Install SDKs: `google-cloud-aiplatform`, `gradio`, `pandas`.

### Document Parsing Layer
- [ ] Build parsers:
  - PDF → `PyMuPDF` / `pdfplumber`.
  - Word → `python-docx`.
  - XML → `xmltodict` / `lxml`.
  - Markup (HTML/JSON-LD) → `BeautifulSoup`.
- [ ] Normalize parsed text into **requirement chunks**.

### AI Test Case Generation
- [ ] Use **Gemini via Vertex AI**.
- [ ] Prompt format:You are a healthcare QA expert.
Convert the following requirement into test cases with:
ID, Title, Preconditions, Steps, Expected Result, Compliance Tag.



- [ ] Parse AI output into **pandas DataFrame**.
- [ ] Add CSV/Excel export.

### Compliance (Lightweight for PoC)
- [ ] Create a **small mapping dictionary** (e.g., `"encryption" → IEC 62304`, `"audit log" → ISO 13485`).
- [ ] Add compliance tags to generated test cases.

### UI – MVP Demo
- [ ] Use **Gradio** for a quick interface:
- Upload file → See test cases → Download CSV.
- [ ] Local test: `demo.launch()`.

### Deployment Options
- **Option 1: Hugging Face Spaces** → easiest for demo (free tier).
- **Option 2: Google Cloud Run (scalable, secure)**:
- [ ] Containerize with **Docker**.
- [ ] Deploy Gradio/FastAPI app on **Cloud Run**.
- [ ] Configure **HTTPS endpoint with IAM** for controlled access.

✅ **Deliverable:** Upload healthcare requirement doc → AI-generated test cases (CSV/Excel) with compliance tags → deployed demo.

---

## Phase 2 – Compliance & Traceability (3–6 Months)

**Goal:** Build audit-ready features.

### Compliance Mapping Expansion
- [ ] Collect **FDA, IEC 62304, ISO 13485, ISO 27001** clauses.
- [ ] Extend AI prompts to link requirements → standards.

### Traceability Matrix
- [ ] Store requirements ↔ test cases ↔ compliance rules in **BigQuery**.
- [ ] Generate **traceability reports** (Excel/PDF).

### Audit Logging
- [ ] Log AI outputs (input requirement, generated case, timestamp, user).
- [ ] Provide **exportable logs** for audits.

✅ **Deliverable:** AI outputs with full compliance + traceability matrix.

---

## Phase 3 – Toolchain Integration (6–12 Months)

**Goal:** Connect with enterprise tools (Jira, Polarion, Azure DevOps).

### Jira Integration
- [ ] Use **Jira REST API** to push test cases.
- [ ] Sync updates back from Jira.

### Polarion Integration
- [ ] Use **Polarion WebServices API**.

### Azure DevOps Integration
- [ ] Use **ADO Test Plans API**.

✅ **Deliverable:** Auto-sync AI test cases with enterprise ALM tools.

---

## Phase 4 – Scale & GDPR Compliance (12+ Months)

**Goal:** Enterprise readiness.

### GDPR Compliance
- [ ] Add **PHI/PII anonymization pipeline**.
- [ ] Encrypt data at rest (**Cloud KMS**).

### Scalability
- [ ] Deploy full stack on **Google Cloud Run + GKE (Kubernetes)**.
- [ ] Add **role-based access** with Firebase Auth.

### Continuous Improvement
- [ ] Fine-tune **Gemini** with healthcare corpora.
- [ ] Add **multi-language support** (for EU).
- [ ] Build **analytics dashboard** (time saved, compliance coverage).

✅ **Deliverable:** Enterprise-grade, secure, GDPR-compliant SaaS.

---

## ⚡ Key Decision for PoC Deployment

- **Hugging Face Spaces** → fastest to show stakeholders.
- **Google Cloud Run** → scalable, secure, enterprise-grade from day 1.

👉 For your evaluation, I’d suggest:
- Build with **Gradio** (fast UI).
- Deploy first on **Hugging Face** for demo.
- Mirror same app on **Google Cloud Run** for security & scalability.
