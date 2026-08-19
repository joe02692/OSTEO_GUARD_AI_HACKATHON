import numpy as np
import re
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
import google.generativeai as genai
import os

# Initialize database path. We use the populated one.
# It seems the notebook saved it in `osteoguard_ai/Code/osteoarthritis_db`.
DB_PATH = os.path.join(os.path.dirname(__file__), "osteoguard_ai", "Code", "osteoarthritis_db")

# Initialize ChromaDB client and collection globally
_chroma_client = None
_collection = None
_bm25 = None
_all_docs = None
_all_ids = None
_all_metadatas = None
_cross_encoder = None

def init_backend():
    global _chroma_client, _collection, _bm25, _all_docs, _all_ids, _all_metadatas, _cross_encoder
    
    if _collection is not None:
        return # Already initialized

    try:
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="NeuML/pubmedbert-base-embeddings")
        _chroma_client = chromadb.PersistentClient(path=DB_PATH)
        _collection = _chroma_client.get_collection(
            name="osteoarthritis_guidelines_v2", 
            embedding_function=sentence_transformer_ef
        )
        
        # Build Keyword (BM25) Index
        all_data = _collection.get(include=["documents", "metadatas"])
        _all_docs = all_data["documents"]
        _all_ids = all_data["ids"]
        _all_metadatas = all_data["metadatas"]
        
        tokenized_corpus = [tokenize_clean(doc) for doc in _all_docs]
        _bm25 = BM25Okapi(tokenized_corpus)
        
        # Load Cross-Encoder Model
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    except Exception as e:
        print(f"Error initializing backend: {e}")
        raise e

def tokenize_clean(text):
    return re.sub(r'\W+', ' ', text).lower().split()

def reciprocal_rank_fusion(semantic_ranks, keyword_ranks, k=60):
    rrf_scores = {}
    for rank, chunk_id in enumerate(semantic_ranks):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    for rank, chunk_id in enumerate(keyword_ranks):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

def reranked_hybrid_search(query, top_n=5, fetch_k=40):
    init_backend()
    
    JUNK_SECTIONS = [
        "Contents", "REFERENCES", "ACKNOWLEDGMENTS", "METHODS", "Overview",
        "Recommendations for research", "Key recommendations for research",
        "Other recommendations for research", "Rationale and impact", "Context",
        "Update information", "Your responsibility"
    ]
    
    semantic_results = _collection.query(query_texts=[query], n_results=fetch_k)
    semantic_ranked_ids = semantic_results["ids"][0]
    
    keyword_scores = _bm25.get_scores(tokenize_clean(query))
    top_keyword_indices = np.argsort(keyword_scores)[::-1][:fetch_k]
    keyword_ranked_ids = [_all_ids[i] for i in top_keyword_indices]
    
    fused_results = reciprocal_rank_fusion(semantic_ranked_ids, keyword_ranked_ids)[:fetch_k]
    
    cross_inp = []
    candidate_chunks = []
    
    for chunk_id, _ in fused_results:
        original_idx = _all_ids.index(chunk_id)
        metadata = _all_metadatas[original_idx]
        text = _all_docs[original_idx]
        
        # Apply the junk filter
        if any(junk.lower() in metadata.get('section_title', '').lower() for junk in JUNK_SECTIONS):
            continue
            
        enriched_text = f"Section: {metadata.get('section_title', '')}. {text}"
        
        cross_inp.append([query, enriched_text])
        candidate_chunks.append((chunk_id, metadata, text))
        
        if len(cross_inp) == 20: 
            break
            
    if not cross_inp:
        return []
        
    cross_scores = _cross_encoder.predict(cross_inp)
    reranked_pairs = sorted(zip(cross_scores, candidate_chunks), key=lambda x: x[0], reverse=True)
    
    return reranked_pairs[:top_n]

def generate_response(query, api_key, top_n=5):
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel("gemini-3.6-flash")
    
    # Translate query to English for search
    translate_prompt = f"Translate the following text to English for a clinical database search. If it is already in English, return it exactly as is. Output ONLY the translated text.\nText: {query}"
    try:
        translated_query = model.generate_content(translate_prompt).text.strip()
    except Exception as e:
        return f"Error translating query: {str(e)}", []
        
    search_results = reranked_hybrid_search(translated_query, top_n=top_n)
    
    if not search_results:
        return "I could not find any relevant information in the guidelines to answer your query.", []
    
    context_text = ""
    sources = []
    for score, chunk in search_results:
        chunk_id, metadata, text = chunk
        doc_name = metadata.get('document_name', 'Unknown Document')
        page_num = metadata.get('page_number', 'N/A')
        source_url = metadata.get('source_url', '#')
        
        context_text += f"--- Source: {doc_name} (Page {page_num}) ---\n{text}\n\n"
        sources.append({
            "doc_name": doc_name,
            "page": page_num,
            "url": source_url,
            "score": float(score)
        })
        
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
    except Exception as e:
        return f"Error generating response: {str(e)}", sources
