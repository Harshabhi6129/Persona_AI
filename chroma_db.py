import math
import numpy as np
from embeddings import get_embedding, load_persona_texts
_is_initialized = False

# Global in-memory lists
_persona_collection = []
_memory_collection = []
_correction_collection = []

def build_chroma_collection():
    """
    In-memory stand-in for a vector DB. We load & embed persona data once.
    """
    global _is_initialized
    if not _is_initialized:
        _init_persona_collection()
        _is_initialized = True
    return None, _persona_collection, _memory_collection, _correction_collection

def _init_persona_collection():
    # Flatten + chunk + embed all persona text
    texts = load_persona_texts()
    for i, text in enumerate(texts):
        emb = get_embedding(text)
        _persona_collection.append({
            "id": f"persona_{i}",
            "text": text,
            "embedding": emb
        })

def _cosine_sim(vec_a, vec_b):
    if not vec_a.any() or not vec_b.any():
        return 0.0
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b + 1e-10)

def search_chroma(query, collection, k=3):
    """
    Basic cosine similarity search in the in-memory collection.
    Returns up to k top 'text' strings.
    """
    if not query.strip():
        return []

    query_emb = get_embedding(query)
    scored = []
    for doc in collection:
        score = _cosine_sim(query_emb, doc["embedding"])
        scored.append((score, doc["text"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_texts = [item[1] for item in scored[:k]]
    return top_texts

def multi_pass_retrieval(query, persona_collection, memory_collection, correction_collection):
    """
    - Check corrections first (k=1)
    - Then gather more persona data (k=5)
    - Then memory context (k=3)
    - Merge results.
    """
    corrected_responses = search_chroma(query, correction_collection, k=1)
    if corrected_responses:
        return f"🔹 **User-Corrected Response:**\n{corrected_responses[0]}"

    persona_context = search_chroma(query, persona_collection, k=5)
    memory_context = search_chroma(query, memory_collection, k=3)

    combined = []
    if memory_context:
        combined.append("🔹 **Past Conversations:**\n" + "\n".join(memory_context))
    if persona_context:
        combined.append("🔹 **Persona Background:**\n" + "\n".join(persona_context))

    return "\n\n".join(combined)

def store_memory(user_query, response):
    """
    Save the latest user query + response for future retrieval.
    """
    emb = get_embedding(user_query)
    _memory_collection.append({
        "id": f"memory_{user_query[:30]}",
        "text": response,
        "embedding": emb
    })

def store_correction(user_query, corrected_response):
    """
    Save a correction snippet for future retrieval.
    """
    emb = get_embedding(user_query)
    _correction_collection.append({
        "id": f"correction_{user_query[:30]}",
        "text": corrected_response,
        "embedding": emb
    })
