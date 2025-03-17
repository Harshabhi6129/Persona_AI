import openai
import streamlit as st
import numpy as np

# Retrieve the API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Define your desired embedding model here
# or read it from st.secrets if you prefer
EMBEDDING_MODEL = "text-embedding-ada-002"

def get_embedding(text, model=EMBEDDING_MODEL):
    """Generate an embedding for the given text using OpenAI's model."""
    response = openai.Embedding.create(
        model=model,
        input=text
    )
    return response["data"][0]["embedding"]

def flatten_dict(d, parent_key=''):
    """
    Recursively extracts text from a nested dictionary
    and converts it into structured text.
    """
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

def load_persona_texts():
    """
    Load persona data from JSON (persona_data.json)
    and extract expanded textual segments for embedding.
    """
    import json
    with open("persona_data.json", "r") as f:
        persona = json.load(f)

    texts = flatten_dict(persona)
    return texts
