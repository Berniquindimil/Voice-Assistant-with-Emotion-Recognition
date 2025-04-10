# Chatbot with Llama 3.2 and Streamlit

This project uses the `llama-3.2-1b-instruct` model through LM Studio, integrating it into a web interface with **Streamlit** to build a conversational chatbot.

## Step 1: Set up LM Studio

1. Download LM Studio from [lmstudio.ai](https://lmstudio.ai).
2. Within LM Studio, download the `llama-3.2-1b-instruct` model.
3. Make sure the LM Studio server is running locally.

## Step 2: Test the model with Python

```python
import lmstudio as lms

model = lms.llm("llama-3.2-1b-instruct")
result = model.respond("What is the meaning of life?")
print(result).
```
The model works good now but in the terminal.

## Step 3: Do the interface with streamlit

I take in account this tutorial of the streamlit documentation: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps. And adapt it with my llama3.2 model downloaded in LM Studio.

```python
import lmstudio as lms
import streamlit as st
import numpy as np

st.title("Emotional chatbot")

# Initialize the model only once
if "model" not in st.session_state:
    st.session_state.model = lms.llm("llama-3.2-1b-instruct")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Response with the llama3.2 model
    response = st.session_state.model.respond(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
```

