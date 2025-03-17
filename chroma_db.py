import numpy as np
from embeddings import get_embedding, load_persona_texts

# We keep the same function names ("build_chroma_collection", etc.)
# so the rest of your code doesn't need to change.

# Global in-memory lists
_INITIALIZED = False
PERSONA_DOCS = []
MEMORY_DOCS = []
CORRECTION_DOCS = []


def init_store():
    """One-time setup: load persona data, create embeddings, store them in PERSONA_DOCS."""
    global _INITIALIZED, PERSONA_DOCS
    if _INITIALIZED:
        return

    # Load and embed persona text lines
    persona_texts = load_persona_texts()  # from embeddings.py
    for text in persona_texts:
        emb = get_embedding(text)
        PERSONA_DOCS.append({"text": text, "embedding": emb})

    _INITIALIZED = True


def build_chroma_collection():
    """
    Mimics returning client and collections, but actually just ensures init_store is called.
    Returns placeholders so existing calls in llm.py won't break.
    """
    init_store()
    client = None
    persona_collection = PERSONA_DOCS
    memory_collection = MEMORY_DOCS
    correction_collection = CORRECTION_DOCS
    return client, persona_collection, memory_collection, correction_collection


def _search_naive(query, docs, k=2):
    """
    Naive similarity search over 'docs' (list of dicts: {"text":..., "embedding":...}).
    Returns up to top k doc texts by cosine similarity.
    """
    query_emb = get_embedding(query)
    sims = []
    for doc in docs:
        doc_emb = doc["embedding"]
        # Cosine similarity
        cos_sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
        sims.append((doc["text"], cos_sim))

    # Sort by similarity descending
    sims.sort(key=lambda x: x[1], reverse=True)
    top_docs = [s[0] for s in sims[:k]]
    return top_docs


def multi_pass_retrieval(query, persona_collection, memory_collection, correction_collection):
    """
    - Searches user-corrected responses FIRST
    - Searches general persona background
    - Searches past conversations separately
    - Merges all results into a final response.
    """
    # 1) Correction search
    corrected = _search_naive(query, correction_collection, k=1)
    if corrected:
        return f"🔹 **User-Corrected Response:**\n{corrected[0]}"

    # 2) Persona search
    persona_context = _search_naive(query, persona_collection, k=3)
    # 3) Memory search
    memory_context = _search_naive(query, memory_collection, k=2)

    combined_context = []
    if memory_context:
        combined_context.append("🔹 **Past Conversations:**\n" + "\n".join(memory_context))
    if persona_context:
        combined_context.append("🔹 **Persona Background:**\n" + "\n".join(persona_context))

    return "\n\n".join(combined_context)


def store_memory(user_query, response):
    """
    Stores a conversation exchange in the in-memory 'MEMORY_DOCS'.
    """
    _, _, memory_collection, _ = build_chroma_collection()
    emb = get_embedding(user_query)
    memory_collection.append({"text": response, "embedding": emb})


def store_correction(user_query, corrected_response):
    """
    Stores user-corrected responses in the in-memory 'CORRECTION_DOCS'.
    """
    _, _, _, correction_collection = build_chroma_collection()
    emb = get_embedding(user_query)
    correction_collection.append({"text": corrected_response, "embedding": emb})
