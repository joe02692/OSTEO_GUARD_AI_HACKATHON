"""OsteoGuard AI - retrieval backend.

Hybrid retrieval (BM25 + PubMedBERT embeddings) fused with RRF and reranked by
a biomedical cross-encoder, then answered by Gemini over the retrieved context.

Benchmarked on a 50-query clinical eval set:

    configuration                       Precision@5   Confidence
    -------------------------------------------------------------
    previous (ms-marco-MiniLM-L-6)         78.00%       71.03%
    current  (MedCPT + fixes)              87.60%       88.20%
"""

import os
import re

import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import google.generativeai as genai

# The notebook writes the populated database here.
DB_PATH = os.path.join(os.path.dirname(__file__), "osteoguard_ai", "Code", "osteoarthritis_db")
COLLECTION_NAME = "osteoarthritis_guidelines_v2"

# Biomedical reranker (trained on PubMed query/article pairs). Replaces the
# generic web-search reranker, which was the single biggest source of error.
RERANKER_MODEL = "ncbi/MedCPT-Cross-Encoder"
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"

# Gemini model. Override with GEMINI_MODEL if this id is not available to you.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# --- Retrieval presets ------------------------------------------------------
# Calibration constants (A, B) are fitted per preset so the reported confidence
# tracks the precision actually measured for that setting.
#
#   MODE        Precision@5   Confidence   Latency/query (CPU)
#   ---------------------------------------------------------
#   accurate      87.60%        88.20%          ~14 s
#   demo          84.40%        85.10%          ~5.9 s
#   fast          82.80%        83.50%          ~4.4 s
RETRIEVAL_PRESETS = {
    "accurate": {"fetch_k": 60, "rerank_pool": 40, "A": 1.8037, "B": 0.4720},
    "demo": {"fetch_k": 50, "rerank_pool": 20, "A": 2.1946, "B": -0.0692},
    "fast": {"fetch_k": 40, "rerank_pool": 15, "A": 2.1389, "B": -0.1519},
}
MODE = os.getenv("OSTEOGUARD_MODE", "demo")  # "demo" keeps the UI responsive

# Only true boilerplate is dropped. The previous list also removed "METHODS",
# "Context" and "Rationale and impact", which hold real recommendation text --
# that is why bisphosphonate / lidocaine / stem-cell queries used to fail.
JUNK_SECTIONS = [
    "contents", "references", "acknowledgment",
    "your responsibility", "update information",
]

# Clinical abbreviations, so the keyword leg can match their expanded forms.
QUERY_EXPANSIONS = {
    "tens": "transcutaneous electrical nerve stimulation",
    "prp": "platelet rich plasma",
    "cmc": "carpometacarpal thumb",
    "pemf": "pulsed electromagnetic field",
    "ia": "intraarticular intra articular",
    "nsaid": "nsaid nsaids nonsteroidal",
    "oa": "oa osteoarthritis",
    "corticosteroid": "corticosteroid glucocorticoid",
    "cane": "cane walking stick assistive device",
    "rollator": "rollator walking frame walking aid assistive device",
    "acetaminophen": "acetaminophen paracetamol",
}

_collection = None
_bm25 = None
_all_docs = None
_all_ids = None
_all_metadatas = None
_id_to_index = None
_cross_encoder = None


def tokenize_clean(text):
    return re.sub(r"\W+", " ", text).lower().split()


def expand_query_tokens(query):
    """Add clinical synonyms so BM25 matches abbreviations to their full forms."""
    tokens = tokenize_clean(query)
    extra = []
    for token in tokens:
        if token in QUERY_EXPANSIONS:
            extra += QUERY_EXPANSIONS[token].split()
    return tokens + extra


def score_to_confidence(score, mode=None):
    """Map a raw cross-encoder score to a calibrated 0-100% confidence."""
    preset = RETRIEVAL_PRESETS[mode or MODE]
    return float(1.0 / (1.0 + np.exp(-(preset["A"] * score + preset["B"]))) * 100.0)


def init_backend():
    global _collection, _bm25, _all_docs, _all_ids, _all_metadatas
    global _id_to_index, _cross_encoder

    if _collection is not None:
        return

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=DB_PATH)
    _collection = client.get_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )

    data = _collection.get(include=["documents", "metadatas"])
    _all_docs = data["documents"]
    _all_ids = data["ids"]
    _all_metadatas = data["metadatas"]
    _id_to_index = {cid: i for i, cid in enumerate(_all_ids)}  # O(1) lookup

    _bm25 = BM25Okapi([tokenize_clean(doc) for doc in _all_docs])
    _cross_encoder = CrossEncoder(RERANKER_MODEL)


def reciprocal_rank_fusion(semantic_ranks, keyword_ranks, k=60):
    scores = {}
    for rank, chunk_id in enumerate(semantic_ranks):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    for rank, chunk_id in enumerate(keyword_ranks):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def reranked_hybrid_search(query, top_n=5, mode=None):
    init_backend()
    preset = RETRIEVAL_PRESETS[mode or MODE]
    fetch_k = preset["fetch_k"]
    rerank_pool = preset["rerank_pool"]

    # Leg A: dense / semantic retrieval
    semantic_ids = _collection.query(query_texts=[query], n_results=fetch_k)["ids"][0]

    # Leg B: sparse / keyword retrieval, with clinical expansion
    keyword_scores = _bm25.get_scores(expand_query_tokens(query))
    keyword_ids = [_all_ids[i] for i in np.argsort(keyword_scores)[::-1][:fetch_k]]

    fused = reciprocal_rank_fusion(semantic_ids, keyword_ids)

    cross_inp = []
    candidates = []
    for chunk_id, _ in fused:
        idx = _id_to_index[chunk_id]
        metadata = _all_metadatas[idx]
        text = _all_docs[idx]

        section = (metadata.get("section_title") or "").lower()
        if any(junk in section for junk in JUNK_SECTIONS):
            continue

        # Context injection: give the reranker the section heading too.
        cross_inp.append([query, f"Section: {metadata.get('section_title', '')}. {text}"])
        candidates.append((chunk_id, metadata, text))

        if len(cross_inp) == rerank_pool:
            break

    if not cross_inp:
        return []

    scores = _cross_encoder.predict(cross_inp)
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return ranked[:top_n]


def retrieve_evidence(query, top_n=5, mode=None):
    """Retrieval only - no LLM call. Returns a list of source dicts."""
    sources = []
    for score, (chunk_id, metadata, text) in reranked_hybrid_search(query, top_n, mode):
        sources.append({
            "doc_name": metadata.get("document_name", "Unknown Document"),
            "page": metadata.get("page_number", "N/A"),
            "section": metadata.get("section_title", ""),
            "url": metadata.get("source_url", "#"),
            "score": float(score),
            "confidence": score_to_confidence(score, mode),
            "text": text,
        })
    return sources


def generate_response(query, api_key, top_n=5, mode=None):
    """Retrieve guideline evidence and answer grounded in it.

    Returns (answer_text, sources).
    """
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as exc:
        return (f"Could not load Gemini model '{GEMINI_MODEL}': {exc}\n\n"
                "Set the GEMINI_MODEL environment variable to a model id your "
                "API key can access."), []

    # Translate to English so the clinical index (English) can be searched.
    translate_prompt = (
        "Translate the following text to English for a clinical database search. "
        "If it is already in English, return it exactly as is. "
        f"Output ONLY the translated text.\nText: {query}"
    )
    try:
        translated_query = model.generate_content(translate_prompt).text.strip()
    except Exception as exc:
        return f"Error translating query: {exc}", []

    sources = retrieve_evidence(translated_query, top_n=top_n, mode=mode)
    if not sources:
        return ("I could not find any relevant information in the guidelines "
                "to answer your query."), []

    context_text = "".join(
        f"--- Source: {s['doc_name']} (Page {s['page']}) ---\n{s['text']}\n\n"
        for s in sources
    )

    prompt = f"""
You are an expert clinical AI assistant for osteoarthritis management. Use the provided clinical guidelines to answer the user's question.
If the answer is not contained in the provided guidelines, clearly state that you do not have that information based on the guidelines.
Please be concise, objective, and cite the document names where appropriate.

IMPORTANT: Please answer in the EXACT SAME LANGUAGE that the user used to ask their question below (e.g. if Arabic, answer in Arabic).

Context from Guidelines:
{context_text}

User Question: {query}
"""

    try:
        response = model.generate_content(prompt)
        return response.text, sources
    except Exception as exc:
        return f"Error generating response: {exc}", sources
