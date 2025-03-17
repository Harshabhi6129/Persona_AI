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

def tokenize_text(text):
    """
    Tokenize the text using tiktoken to measure length.
    Returns a list of tokens.
    """
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    return enc.encode(text)

def chunk_text(text, max_tokens=300, overlap=50):
    """
    Splits text into overlapping chunks of up to 'max_tokens' tokens each.
    Overlap helps preserve context between chunks.
    """
    tokens = tokenize_text(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        # Convert tokens back to string
        chunk_text = tiktoken.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens - overlap  # move forward with overlap
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
            # Convert lists to text
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
    # Next, chunk any large segments
    chunked_texts = []
    for text in flat_texts:
        # Chunk the text if it’s long
        sub_chunks = chunk_text(text, max_tokens=300, overlap=50)
        chunked_texts.extend(sub_chunks)
    return chunked_texts
