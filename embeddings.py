import openai
import streamlit as st
import numpy as np
import json
import tiktoken

openai.api_key = st.secrets["OPENAI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-ada-002"


def get_embedding(text, model=EMBEDDING_MODEL):
    """
    Generate a vector embedding for the given text using OpenAI's model.
    If the text is empty or whitespace, return a zero vector.
    """
    text = text.strip()
    if not text:
        return np.zeros(1536, dtype=np.float32)
    response = openai.Embedding.create(model=model, input=text)
    emb = response["data"][0]["embedding"]
    return np.array(emb, dtype=np.float32)


def chunk_text(text, max_tokens=300, overlap=50):
    """
    Splits text into overlapping chunks of up to 'max_tokens' tokens each.
    Overlap helps preserve context between chunks.

    We use the same encoder (enc) for both encoding and decoding.
    """
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = enc.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_str = enc.decode(chunk_tokens)  # Decode with the same encoder
        chunks.append(chunk_str)
        start += max_tokens - overlap  # Move forward with overlap

    return chunks


def flatten_dict(d, parent_key=''):
    """
    Recursively extracts text from a nested dictionary
    and converts it into structured text segments.
    """
    texts = []
    for key, value in d.items():
        new_key = f"{parent_key} {key}".strip() if parent_key else key
        if isinstance(value, dict):
            texts.extend(flatten_dict(value, new_key))
        elif isinstance(value, list):
            joined_list = ", ".join(str(v) for v in value)
            texts.append(f"{new_key}: {joined_list}")
        else:
            texts.append(f"{new_key}: {value}")
    return texts


def load_persona_texts():
    """
    Load persona data from persona_data.json and flatten it to text segments.
    Then chunk any long segments for more precise retrieval.
    """
    with open("persona_data.json", "r", encoding="utf-8") as f:
        persona = json.load(f)

    flat_texts = flatten_dict(persona)
    chunked_texts = []

    for text in flat_texts:
        # Chunk the text if it’s long
        sub_chunks = chunk_text(text, max_tokens=300, overlap=50)
        chunked_texts.extend(sub_chunks)

    return chunked_texts
