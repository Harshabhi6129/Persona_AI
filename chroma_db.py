import math
import numpy as np
from embeddings import get_embedding

# Global in-memory lists for persona, memory, corrections
_persona_collection = []
_memory_collection = []
_correction_collection = []
_is_initialized = False

def build_chroma_collection():
    """
    Initializes the in-memory collections once, returning references
    so that old code continues to work as expected.
    """
    global _is_initialized
    if not _is_initialized:
        _init_persona_collection()
        _is_initialized = True
    # We return "client" as None because there's no real DB client
    return None, _persona_collection, _memory_collection, _correction_collection

def _init_persona_collection():
    """
    Load persona data from `persona_data.json`, flatten it,
    embed each chunk, and store in _persona_collection.
    """
    from embeddings import load_persona_texts
    texts = load_persona_texts()
    for i, text in enumerate(texts):
        emb = get_embedding(text)
        _persona_collection.append({
            "id": f"persona_{i}",
            "text": text,
            "embedding": emb
        })

def search_chroma(query, collection, k=2):
    """
    Perform a naive cosine similarity search in the given in-memory collection.
    Returns up to 'k' top documents' text.
    """
    if not query.strip():
        return []

    query_emb = get_embedding(query)
    
    # Calculate cosine similarity with each doc in the collection
    scored = []
    for doc in collection:
        score = _cosine_sim(query_emb, doc["embedding"])
        scored.append((score, doc["text"]))

    # Sort by descending similarity, return top k doc texts
    scored.sort(key=lambda x: x[0], reverse=True)
    top_texts = [item[1] for item in scored[:k]]
    return top_texts

def multi_pass_retrieval(query, persona_collection, memory_collection, correction_collection):
    """
    Replicates the old multi-pass logic:
    - Look for corrections first
    - Then persona background
    - Then memory
    - Combine the results
    """
    # 🔹 Retrieve user corrections FIRST
    corrected_responses = search_chroma(query, correction_collection, k=1)
    if corrected_responses:
        return f"🔹 **User-Corrected Response:**\n{corrected_responses[0]}"

    # 🔹 Otherwise, retrieve persona data + memory
    persona_context = search_chroma(query, persona_collection, k=3)
    memory_context = search_chroma(query, memory_collection, k=2)

    combined_context = []
    if memory_context:
        combined_context.append("🔹 **Past Conversations:**\n" + "\n".join(memory_context))
    if persona_context:
        combined_context.append("🔹 **Persona Background:**\n" + "\n".join(persona_context))

    return "\n\n".join(combined_context)

def store_memory(user_query, response):
    """
    Stores a conversation exchange (the response) in _memory_collection.
    """
    emb = get_embedding(user_query)
    _memory_collection.append({
        "id": f"memory_{user_query[:30]}",
        "text": response,
        "embedding": emb
    })

def store_correction(user_query, corrected_response):
    """
    Stores user-corrected responses in _correction_collection.
    """
    emb = get_embedding(user_query)
    _correction_collection.append({
        "id": f"correction_{user_query[:30]}",
        "text": corrected_response,
        "embedding": emb
    })

def _cosine_sim(vec_a, vec_b):
    """
    Compute the cosine similarity between two embedding vectors.
    """
    if not vec_a.any() or not vec_b.any():
        return 0.0
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b + 1e-10)
