import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the fine-tuned model and tokenizer
model_path = "./llama-therapy-lora"  # Path to your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Select the appropriate device for Apple Silicon (M1, M2...)
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Metal Performance Shader
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Move the model to the selected device
model.to(device)

# System prompt to guide the chatbot's personality and tone
system_prompt = (
    "You are a compassionate therapist helping people deal with emotional challenges. "
    "Always respond with empathy and open-ended questions that encourage reflection.\n\n"
)

# Set up the Streamlit interface
st.set_page_config(page_title="Therapy Chatbot", page_icon="💬")
st.title("💬 Therapy Chatbot")
st.write("A fine-tuned TinyLLaMA model for empathetic conversations.")

# Initialize the session state to store the conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Text input for the user message
user_input = st.text_input("You:", key="input")

# Generate and display the chatbot's response
if user_input:
    # Add the user's message to the conversation history
    st.session_state.history.append(f"User: {user_input}")

    # Construct the prompt using system prompt and conversation history
    prompt = system_prompt + "\n".join(st.session_state.history) + "\nTherapist:"

    # Tokenize and encode the prompt
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(device)

    # Generate a response using the model
    output_ids = model.generate(
        input_ids,
        max_new_tokens=150,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    # Decode and clean the model's output
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = response.split("Therapist:")[-1].strip()

    # Add the chatbot's response to the history
    st.session_state.history.append(f"Therapist: {response}")

    # Display the last few messages in the conversation
    for line in st.session_state.history[-6:]:
        if line.startswith("User:"):
            st.markdown(f"**{line}**")
        else:
            st.markdown(f"{line}")

# Button to reset the conversation history
if st.button("🔄 Reset conversation"):
    st.session_state.history = []
