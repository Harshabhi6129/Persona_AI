import openai
import streamlit as st

# Retrieve the API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# A default embedding model (not used in this no-DB approach)
EMBEDDING_MODEL = "text-embedding-ada-002"

def get_embedding(text, model=EMBEDDING_MODEL):
    """Generate an embedding for the given text using OpenAI's model."""
    response = openai.Embedding.create(
        model=model,
        input=text
    )
    return response["data"][0]["embedding"]

# Additional helper if needed, or remove entirely
def flatten_dict(d, parent_key=''):
    texts = []
    for key, value in d.items():
        new_key = f"{parent_key} {key}".strip() if parent_key else key
        if isinstance(value, dict):
            texts.extend(flatten_dict(value, new_key))
        elif isinstance(value, list):
            texts.append(f"{new_key}: " + ", ".join(str(v) for v in value))
        else:
            texts.append(f"{new_key}: {value}")
    return texts
