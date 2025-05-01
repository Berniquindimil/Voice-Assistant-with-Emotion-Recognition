import lmstudio as lms
import streamlit as st
import numpy as np

st.title("Emotional chatbot")

# Load the model once
if "model" not in st.session_state:
    st.session_state.model = lms.llm("llama-3.2-1b-instruct")

# Add the hidden system prompt only once
SYSTEM_PROMPT = (
    "You are a professional therapist specializing in mental health. "
    "You listen with empathy, validate emotions, and offer guidance without judgment. "
    "You respond in a kind, clear, and focused manner focused on emotional well-being."
)

if "messages" not in st.session_state:
    st.session_state.messages = []  # Visible messages
if "full_context" not in st.session_state:
    # Internal conversation for the model, includes the system prompt
    st.session_state.full_context = [{"role": "system", "content": SYSTEM_PROMPT}]

# Display visible messages only
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user input
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.full_context.append({"role": "user", "content": prompt})

    # Build full prompt for model
    prompt_text = ""
    for msg in st.session_state.full_context:
        if msg["role"] == "system":
            prompt_text += f"{msg['content']}\n"
        elif msg["role"] == "user":
            prompt_text += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt_text += f"Therapist: {msg['content']}\n"

    prompt_text += "Therapist:"

    # Get model response
    response = st.session_state.model.respond(prompt_text)

    # Show and store response
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.full_context.append({"role": "assistant", "content": response})
