from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Load dataset
dataset = load_dataset("LuangMV97/Empathetic_counseling_Dataset")

# Load model and tokenizer (ensure you're loading the right model)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Apply LoRA (5) for efficient fine-tuning
lora_config = LoraConfig(
    r=4,    # Rank of the decomposition factor in LoRA matrices. Instead of updating the entire weights matrix, it adds two smaller matrices.
    lora_alpha=16,  # Scaling factor that controls how much the LoRA update affects the model. 16/4 = 4 and is a good balance between adaptation and stability
    target_modules=["q_proj", "v_proj"],    # Defines which layers of the model LoRA will be applied to. query projection and value projection because are very important for the attention mechanism.
    lora_dropout=0.1,   # Dropout layers for avoiding overfitting
    bias="none",    # no bias are updated.
    task_type=TaskType.CAUSAL_LM    # self-generating text generation
)
model = get_peft_model(model, lora_config)

# Tokenize dataset
def tokenize_function(examples):
    # Combine input and label into a dialogue-style prompt
    prompts = [f"User: {inp}\nTherapist: {resp}" for inp, resp in zip(examples["input"], examples["label"])]
    
    # Tokenize with truncation and padding to ensure uniform input size
    tokenized = tokenizer(prompts, truncation=True, padding="max_length", max_length=256)

    # Set the input_ids as labels for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=dataset["train"].column_names  # Remove raw text columns to avoid tensor conversion issues
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./llama-therapy-lora",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

# Train the model
#trainer.train()

# Mostrar algunos ejemplos del dataset ya tokenizado
for i in range(3):
    example = tokenized_dataset["train"][i]
    decoded_input = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
    decoded_label = tokenizer.decode(example["labels"], skip_special_tokens=True)

    print(f"\n--- Example {i+1} ---")
    print(f"Decoded Input:\n{decoded_input}")
    print(f"Decoded Label:\n{decoded_label}")


# Save the model and tokenizer
#model.save_pretrained("./llama-therapy-lora")
#tokenizer.save_pretrained("./llama-therapy-lora")
