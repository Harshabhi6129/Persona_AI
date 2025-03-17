import openai
import streamlit as st
import numpy as np
import json

# Use Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-ada-002"

def get_embedding(text, model=EMBEDDING_MODEL):
    """Generate a vector embedding for the given text using OpenAI's model."""
    if not text.strip():
        # Return a zero vector if text is empty
        return np.zeros(1536, dtype=np.float32)
    response = openai.Embedding.create(
        model=model,
        input=text
    )
    emb = response["data"][0]["embedding"]
    return np.array(emb, dtype=np.float32)

def flatten_dict(d, parent_key=''):
    """
    Recursively extracts text from a nested dictionary and
    converts it into structured text segments.
    """
    texts = []
    for key, value in d.items():
        new_key = f"{parent_key} {key}".strip() if parent_key else key
        if isinstance(value, dict):
            texts.extend(flatten_dict(value, new_key))
        elif isinstance(value, list):
            # Convert lists to text
            texts.append(f"{new_key}: " + ", ".join(str(v) for v in value))
        else:
            texts.append(f"{new_key}: {value}")
    return texts

def load_persona_texts():
    """
    Load persona data from persona_data.json and flatten it to text chunks.
    """
    with open("persona_data.json", "r") as f:
        persona = json.load(f)
    texts = flatten_dict(persona)
    return texts
