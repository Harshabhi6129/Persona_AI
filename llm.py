import openai
import streamlit as st
from chroma_db import build_chroma_collection, multi_pass_retrieval, store_memory, store_correction

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
        messages=[{"role": "system", "content": "You are a classifier that determines the appropriate conversational tone."},
                  {"role": "user", "content": personality_prompt}]
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
        messages=[{"role": "system", "content": "You are an AI sentiment analyzer."},
                  {"role": "user", "content": sentiment_prompt}]
    )

    sentiment = response.choices[0].message.content.strip()

    if sentiment not in ["Happy", "Sad", "Angry", "Neutral", "Excited"]:
        return "Neutral"

    return sentiment

def is_basic_conversation(query):
    """
    Determines if the query is a basic greeting or common small talk.
    Returns True for simple conversations, False otherwise.
    """
    basic_phrases = ["hi", "hello", "hey", "good morning", "good evening", "how are you"]
    return query.lower().strip() in basic_phrases

def generate_response(query):
    """
    Retrieve relevant context (via multi_pass_retrieval) and produce a response with 
    adjusted verbosity based on query complexity.
    """
    # Check if the query is a basic greeting
    if is_basic_conversation(query):
        simple_responses = [
            "Hey! How’s it going?",
            "Hi there!",
            "Hello! How can I help?",
            "Hey! What’s on your mind?",
        ]
        return simple_responses[0]  # Return a short response

    # Initialize the in-memory "collections"
    _, persona_collection, memory_collection, correction_collection = build_chroma_collection()

    personality_mode = detect_personality_mode(query)
    sentiment = detect_sentiment(query)

    # Retrieve context
    context = multi_pass_retrieval(query, persona_collection, memory_collection, correction_collection)

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

    # Adjust verbosity dynamically
    if len(query.split()) < 5:
        response_style = "Keep it short and direct."
    elif len(query.split()) < 15:
        response_style = "Answer clearly with some context."
    else:
        response_style = "Provide a detailed response using past experiences."

    prompt = f"""
    You are John Doe, a thoughtful, empathetic, and articulate individual with a detailed life story.
    Your background, experiences, and values have shaped you into a person who is reflective and honest.

    🔹 **Context for this Conversation:**
    {context}

    🔹 **Personality Mode Detected:** {personality_mode}
    🔹 **User Sentiment Detected:** {sentiment}
    
    {mode_instructions[personality_mode]}
    {sentiment_responses[sentiment]}
    {response_style}

    Now, answer the question as if you are John Doe, maintaining consistency with your past statements.

    Question: {query}
    Answer (respond naturally, with consistency and personality):
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are John Doe, a highly personalized AI."},
                  {"role": "user", "content": prompt}]
    )
    final_response = response.choices[0].message.content

    # Store to memory for future context
    store_memory(query, final_response)

    return final_response
