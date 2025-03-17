import chromadb
from embeddings import get_embedding

def build_chroma_collection():
    """
    Build a Chroma collection from persona embeddings.
    """
    client = chromadb.PersistentClient(path="./chroma_db")

    # Main Persona Collection
    persona_collection = client.get_or_create_collection(name="persona_collection")

    # Memory Collection (for past conversations)
    memory_collection = client.get_or_create_collection(name="memory_collection")

    # Correction Collection (NEW - for user feedback)
    correction_collection = client.get_or_create_collection(name="correction_collection")

    return client, persona_collection, memory_collection, correction_collection

def search_chroma(query, collection, k=2):
    """
    Searches the Chroma collection for the most relevant persona context.
    Returns the top 'k' relevant snippets.
    """
    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    return results["documents"][0] if results["documents"] else []

def multi_pass_retrieval(query, persona_collection, memory_collection, correction_collection):
    """
    Implements multi-pass retrieval:
    - Searches user-corrected responses FIRST
    - Searches general persona background
    - Searches past conversations separately
    - Merges all results into a final response.
    """

    # 🔹 Retrieve user corrections FIRST
    corrected_responses = search_chroma(query, correction_collection, k=1)

    # 🔹 If there is a user-approved correction, return that immediately
    if corrected_responses:
        return f"🔹 **User-Corrected Response:**\n{corrected_responses[0]}"

    # 🔹 Otherwise, retrieve persona and memory data
    persona_context = search_chroma(query, persona_collection, k=3)
    memory_context = search_chroma(query, memory_collection, k=2)

    # 🔹 Merge context intelligently
    combined_context = []
    if memory_context:
        combined_context.append("🔹 **Past Conversations:**\n" + "\n".join(memory_context))
    if persona_context:
        combined_context.append("🔹 **Persona Background:**\n" + "\n".join(persona_context))

    return "\n\n".join(combined_context)

def store_memory(user_query, response):
    """
    Stores a conversation exchange in ChromaDB.
    """
    client, _, memory_collection, _ = build_chroma_collection()

    memory_embedding = get_embedding(user_query)
    memory_collection.add(
        documents=[response],
        embeddings=[memory_embedding],
        ids=[f"memory_{user_query[:30]}"]  # Unique ID based on the query
    )

def store_correction(user_query, corrected_response):
    """
    Stores user-corrected responses in ChromaDB.
    """
    client, _, _, correction_collection = build_chroma_collection()

    correction_embedding = get_embedding(user_query)
    correction_collection.add(
        documents=[corrected_response],
        embeddings=[correction_embedding],
        ids=[f"correction_{user_query[:30]}"]  # Unique ID based on the query
    )
