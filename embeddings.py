import openai
import json
import numpy as np
from config import OPENAI_API_KEY, EMBEDDING_MODEL

openai.api_key = OPENAI_API_KEY

def get_embedding(text, model=EMBEDDING_MODEL):
    """Generate embedding for a given text using OpenAI's latest API."""
    response = openai.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

def flatten_dict(d, parent_key=''):
    """
    Recursively extracts text from a nested dictionary and converts it into structured text.
    """
    texts = []
    for key, value in d.items():
        new_key = f"{parent_key} {key}".strip() if parent_key else key
        if isinstance(value, dict):
            texts.extend(flatten_dict(value, new_key))  # Recursively process sub-dictionaries
        elif isinstance(value, list):
            texts.append(f"{new_key}: " + ", ".join(str(v) for v in value))  # Convert lists to text
        else:
            texts.append(f"{new_key}: {value}")  # Store text as key-value pair
    return texts

def load_persona_texts():
    """Load persona data and extract expanded textual segments for embedding."""
    with open("persona_data.json", "r") as f:
        persona = json.load(f)

    texts = flatten_dict(persona)  # Flatten the JSON structure into a list of text strings
    return texts
