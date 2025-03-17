import os
import sys

#
# Force ChromaDB to use DuckDB, not SQLite
# Also override sqlite3 with pysqlite3 if present
#
os.environ["CHROMA_DB_IMPL"] = "duckdb"

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    # pysqlite3 not available, we proceed without it
    pass

import chromadb
from embeddings import get_embedding

# Create Chroma client with DuckDB backend
chroma_client = chromadb.PersistentClient(path="./chroma_db", settings={"database_impl": "duckdb"})

def build_chroma_collection():
    """
    Build or retrieve Chroma collections.
    """
    persona_collection = chroma_client.get_or_create_collection("persona_collection")
    memory_collection = chroma_client.get_or_create_collection("memory_collection")
    correction_collection = chroma_client.get_or_create_collection("correction_collection")

    return chroma_client, persona_collection, memory_collection, correction_collection

def search_chroma(query, collection, k=2):
    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results["documents"][0] if results["documents"] else []

def multi_pass_retrieval(query, persona_collection, memory_collection, correction_collection):
    # 1) Check user corrections
    corrected_responses = search_chroma(query, correction_collection, k=1)
    if corrected_responses:
        return f"🔹 **User-Corrected Response:**\n{corrected_responses[0]}"

    # 2) Search persona & memory
    persona_context = search_chroma(query, persona_collection, k=3)
    memory_context = search_chroma(query, memory_collection, k=2)

    combined_context = []
    if memory_context:
        combined_context.append("🔹 **Past Conversations:**\n" + "\n".join(memory_context))
    if persona_context:
        combined_context.append("🔹 **Persona Background:**\n" + "\n".join(persona_context))

    return "\n\n".join(combined_context)

def store_memory(user_query, response):
    client, _, memory_collection, _ = build_chroma_collection()
    memory_embedding = get_embedding(user_query)
    memory_collection.add(
        documents=[response],
        embeddings=[memory_embedding],
        ids=[f"memory_{user_query[:30]}"]
    )

def store_correction(user_query, corrected_response):
    client, _, _, correction_collection = build_chroma_collection()
    correction_embedding = get_embedding(user_query)
    correction_collection.add(
        documents=[corrected_response],
        embeddings=[correction_embedding],
        ids=[f"correction_{user_query[:30]}"]
    )
