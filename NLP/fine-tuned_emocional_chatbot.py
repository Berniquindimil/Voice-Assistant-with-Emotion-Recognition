from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# Path where your LoRA fine-tuned model is saved
peft_model_path = "./llama-therapy-lora"

# Load LoRA configuration to retrieve the base model name
peft_config = PeftConfig.from_pretrained(peft_model_path)

# Load the base model (e.g., TinyLlama) using the config
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load the fine-tuned LoRA adapter weights on top of the base model
model = PeftModel.from_pretrained(base_model, peft_model_path)

# Load tokenizer that matches the base model
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

# Define input text (simulating a user prompt)
input_text = "I'm feeling really lost and overwhelmed lately."

# Tokenize input and move it to the same device as the model
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Generate a response from the model using sampling
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    repetition_penalty=1.2
)

# Decode and print the output text, skipping special tokens
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
