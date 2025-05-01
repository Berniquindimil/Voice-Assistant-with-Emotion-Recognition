import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the fine-tuned model and tokenizer
model_path = "/Users/bernardoquindimil/Code/Berniquindimil/Proyect/NLP/Fine-tuned_llm_models/dialogpt-therapy-lora"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

if torch.backends.mps.is_available():
    device = torch.device("mps")
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
st.write("A fine-tuned DialoGPT model for empathetic conversations.")

# Initialize the session state to store the conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Text input for the user message
user_input = st.text_input("You:", key="input")

# Generate and display the chatbot's response
if user_input:
    # Append user message to history
    st.session_state.history.append(f"User: {user_input}")

    # Build the full prompt including system instruction + conversation
    prompt = system_prompt + "\n".join(st.session_state.history) + "\nTherapist:"

    # Tokenize the prompt, returning tensors and attention mask
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="longest"   # pad to the longest sequence in the batch
    ).to(device)

    # Generate a response, **passing in** input_ids and attention_mask
    output_ids = model.generate(
        input_ids=inputs.input_ids,              # <— key fix
        attention_mask=inputs.attention_mask,    # <— key fix
        max_new_tokens=100,
        do_sample=True,
        temperature=0.6,
        top_p=0.85,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id      # ensure padding token is set
    )

    # Decode the full output and split off only the reply
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = full_text.rsplit("Therapist:", 1)[-1].strip()

    # Add assistant’s reply to history and display
    st.session_state.history.append(f"Therapist: {response}")
    for line in st.session_state.history[-6:]:
        if line.startswith("User:"):
            st.markdown(f"**{line}**")
        else:
            st.markdown(line)


# Button to reset the conversation history
if st.button("🔄 Reset conversation"):
    st.session_state.history = []
