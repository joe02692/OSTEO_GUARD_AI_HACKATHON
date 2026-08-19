# OsteoGuard AI / OA ClinicAI

Clinical decision support for osteoarthritis management, grounded in NICE NG226
and the ACR/AF 2019 osteoarthritis guideline.

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**

   ```bash
   python -m streamlit run app_dashboard.py
   ```

   Run it from inside `AI_Hackathon_OsteoGuard-AI-main/` so that
   `.streamlit/config.toml` (which pins the light clinical theme) is picked up.

3. **API key**

   A Gemini API key is required. Either paste it into the sidebar, or create a
   `.env` file next to the app containing `GEMINI_API_KEY=your_key_here`.

   If generation fails with a model error, set `GEMINI_MODEL` to a model id your
   key can access, for example:

   ```bash
   set GEMINI_MODEL=gemini-2.5-flash
   ```

## What it does

**Clinical Assistant** — ask an osteoarthritis management question in any
language. The question is translated for search, answered only from retrieved
guideline passages, and every passage is shown with its document, page and
calibrated confidence.

**Report Summary** — upload a PDF or text clinical report (or paste it) and get
a structured summary under fixed headings: findings, diagnoses, current
management, flagged items, follow-up. The prompt forbids inference, so anything
the report does not state comes back as *not stated in the report* rather than
being filled in. Guideline evidence matching the report's topic is retrieved
separately and labelled as guideline text.

**Statistics** — retrieval performance and live corpus composition read from the
vector store.

## Architecture

- **Frontend**: Streamlit (`app_dashboard.py`, styled by `theme.css`)
- **Retrieval**: BM25 keyword search with clinical abbreviation expansion, run
  alongside `NeuML/pubmedbert-base-embeddings` dense search in ChromaDB, fused
  with reciprocal rank fusion and reranked by `ncbi/MedCPT-Cross-Encoder`
- **Generation**: Google Gemini, answering strictly over retrieved context
- **Data**: NICE NG226 and ACR/AF 2019, 238 indexed passages

Measured on a 50-query clinical evaluation set:

| Configuration | Precision@5 | Confidence |
|---|---|---|
| Previous (ms-marco-MiniLM-L-6) | 78.0% | 71.0% |
| Current (MedCPT + fixes) | **87.6%** | **88.2%** |

## Files

| File | Role |
|---|---|
| `app_dashboard.py` | The application |
| `theme.css` | Clinical green / blue / white theme |
| `backend.py` | Retrieval, answering, report summarisation |
| `reports.py` | PDF and text extraction from uploaded reports |
| `risk.py` | Rule-based OA risk factor display |
| `app.py` | Earlier single-page Q&A UI, kept for reference |
| `osteoguard_ai/Code/Rag_model.ipynb` | Corpus build and evaluation notebook |

## Limits

- **No imaging analysis.** The system does not read X-rays and does not produce
  Kellgren–Lawrence grades. The earlier simulated X-ray module was removed.
- **No OCR.** Scanned PDFs without a text layer cannot be read.
- **Two guidelines only**, including nothing published since them.
- **No storage.** Nothing is persisted between sessions.

A decision-support aid for clinicians. It does not diagnose, does not prescribe,
and does not replace professional medical judgement.
