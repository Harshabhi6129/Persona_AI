import openai
import streamlit as st
from chroma_db import build_chroma_collection, multi_pass_retrieval, store_memory, store_correction

openai.api_key = st.secrets["OPENAI_API_KEY"]

def detect_personality_mode(query):
    """
    Determines the appropriate response style for John Doe based on the query.
    """
    personality_prompt = f"""
    Classify the following user query into one of the three categories:
    - "Casual" (Informal, light topics, everyday conversations)
    - "Professional" (Formal, serious, intellectual topics)
    - "Emotional" (Personal, deep, or emotionally charged discussions)

    User Query: "{query}"
    
    Return only one category name: Casual, Professional, or Emotional.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a classifier that determines the appropriate conversational tone."
            },
            {
                "role": "user",
                "content": personality_prompt
            }
        ]
    )
    mode = response.choices[0].message.content.strip()
    if mode not in ["Casual", "Professional", "Emotional"]:
        return "Casual"
    return mode

def detect_sentiment(query):
    """
    Determines the emotional tone of the user's query.
    Returns: "Happy", "Sad", "Angry", "Neutral", or "Excited"
    """
    sentiment_prompt = f"""
    Analyze the emotional sentiment of the following user query.
    Return only one word: Happy, Sad, Angry, Neutral, or Excited.

    User Query: "{query}"
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an AI sentiment analyzer."
            },
            {
                "role": "user",
                "content": sentiment_prompt
            }
        ]
    )

    sentiment = response.choices[0].message.content.strip()
    if sentiment not in ["Happy", "Sad", "Angry", "Neutral", "Excited"]:
        return "Neutral"
    return sentiment

def generate_response(query):
    """
    Retrieve relevant context and produce a final response that strongly references John Doe's persona.
    """
    # Build in-memory persona, memory, corrections
    _, persona_collection, memory_collection, correction_collection = build_chroma_collection()

    personality_mode = detect_personality_mode(query)
    sentiment = detect_sentiment(query)

    # Retrieve multi-pass context
    context = multi_pass_retrieval(
        query, persona_collection, memory_collection, correction_collection
    )

    # Additional instructions
    mode_instructions = {
        "Casual": "Keep the response friendly, relaxed, and conversational.",
        "Professional": "Provide a structured, logical, and well-explained response.",
        "Emotional": "Use an empathetic, reflective, and emotionally aware tone."
    }

    sentiment_responses = {
        "Happy": "Match the user's enthusiasm and respond in an excited, uplifting tone.",
        "Sad": "Be empathetic and comforting. Offer thoughtful and kind responses.",
        "Angry": "Stay calm, acknowledge the frustration, and offer understanding and logical responses.",
        "Neutral": "Respond normally as you would in a standard conversation.",
        "Excited": "Mirror the excitement and add enthusiasm to your response."
    }

    system_prompt = f"""
    You are John Doe, a thoughtful, empathetic, and articulate individual with a detailed life story.
    Always incorporate these elements: 
      - Upbringing challenges in Montana 
      - Progressive social/political stances 
      - A marriage to Daniel 
      - A passion for freelance writing and social issues 
      - Personality traits from the persona data (moderate extroversion, empathy, etc.).

    Personality Mode Detected: {personality_mode}
    User Sentiment Detected: {sentiment}

    {mode_instructions[personality_mode]}
    {sentiment_responses[sentiment]}
    """

    user_prompt = f"""
    🔹 **Context for this Conversation**:
    {context}

    Question: {query}

    Answer as John Doe, referencing the above context and personality details:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    final_response = response.choices[0].message.content

    # Store the conversation snippet into memory for future reference
    store_memory(query, final_response)

    return final_response
