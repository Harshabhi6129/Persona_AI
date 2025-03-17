import streamlit as st
from llm import generate_response
from chroma_db import store_correction

st.set_page_config(page_title="John Doe - AI Persona", layout="wide")

# 🎨 Sidebar for previous chat history
st.sidebar.title("📜 Previous Chats")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Display previous chats
if len(st.session_state.messages) > 0:
    selected_chat = st.sidebar.selectbox(
        "Select a past message:",
        [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"],
        index=None,
        placeholder="Choose a previous message..."
    )
    if selected_chat:
        st.session_state.selected_chat = selected_chat  # Store selected message

st.title("💬 Chat with John Doe – AI Persona")

st.markdown("""
Welcome to the **Persona-Based Generative Agent**.
John Doe has a detailed life story and memory, allowing him to answer questions as if he were a real person.
Type your message below and start a conversation!
""")

# Display previous messages in the main chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field
user_input = st.chat_input("Type your message...")

# If a previous chat is selected, prefill the input box
if "selected_chat" in st.session_state and st.session_state.selected_chat:
    user_input = st.session_state.selected_chat
    st.session_state.selected_chat = None  # Reset after use

if user_input:
    # Store user message in session
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.spinner("John Doe is thinking..."):
        response = generate_response(user_input)

    # Store John Doe's response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    # Feedback section (user must select one)
    feedback_key = f"feedback_{len(st.session_state.messages)}"
    if feedback_key not in st.session_state:
        feedback = st.radio(
            "Was this response accurate?",
            ["👍 Yes", "👎 No"],
            index=None,  # No default selection
            horizontal=True,
            key=feedback_key
        )

        if feedback:
            st.session_state[feedback_key] = feedback  # Record feedback
            st.experimental_rerun()  # Hide feedback section after selection
