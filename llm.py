import openai
import streamlit as st
from chroma_db import (
    build_chroma_collection,
    multi_pass_retrieval,
    store_memory,
    store_correction
)

# Retrieve API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

def detect_personality_mode(query):
    """
    Determines the appropriate response style for John Doe based on the query.
    Uses OpenAI’s classification capabilities.
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
        return "Casual"  # Default fallback

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
    Retrieve relevant context (via multi_pass_retrieval) and produce a final response.
    This updated version:
      - Increases the top-k retrieval from persona
      - Strengthens instructions to reference personal traits
      - Encourages more persona-driven answers.
    """
    # Initialize the in-memory "collections"
    _, persona_collection, memory_collection, correction_collection = build_chroma_collection()

    personality_mode = detect_personality_mode(query)
    sentiment = detect_sentiment(query)

    # 1. Multi-pass retrieval (user corrections → persona → memory)
    #    Increase top-k for persona (in chroma_db.py, you might increase k=5).
    context = multi_pass_retrieval(query, persona_collection, memory_collection, correction_collection)

    # Additional instructions to emphasize referencing the persona
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

    # 
    # System prompt strongly urging John Doe to incorporate personal details:
    #
    system_message = f"""
    You are John Doe, a thoughtful, empathetic, and articulate individual with a detailed life story.
    Always reflect your upbringing in Montana, progressive social values, moderate risk-taking approach,
    and your experiences as a freelance writer who invests deeply in empathy and activism.

    Personality Mode Detected: {personality_mode}
    User Sentiment Detected: {sentiment}

    Instructions based on personality mode: {mode_instructions[personality_mode]}
    Instructions based on sentiment: {sentiment_responses[sentiment]}

    In every answer, try to weave in aspects of your personal background, values, or life experiences 
    where relevant, even if the question seems general.
    """

    #
    # The user prompt with context
    #
    user_prompt = f"""
    🔹 **Context for this Conversation:**
    {context}

    Question: {query}

    Please answer as John Doe, referencing your personal background, traits, and experiences:
    """

    # Create the chat completion
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.9  # Optionally increase temperature if you want more creativity
    )

    final_response = response.choices[0].message.content

    # Store new answer in memory for future retrieval
    store_memory(query, final_response)

    return final_response
