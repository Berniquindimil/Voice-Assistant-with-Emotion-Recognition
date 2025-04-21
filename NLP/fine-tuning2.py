import math
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

# Load dataset
dataset = load_dataset("LuangMV97/Empathetic_counseling_Dataset")

# Load DialoGPT-medium + pad token
model_id = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Base model + LoRA
base = AutoModelForCausalLM.from_pretrained(model_id)
lora = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(base, lora)

# Tokenize + mask prompt tokens
def tokenize_function(examples):
    texts = [
        f"User: {u}\nTherapist: {r}"
        for u, r in zip(examples["input"], examples["label"])
    ]
    tok = tokenizer(texts, truncation=True, padding="max_length", max_length=256)
    input_ids = tok["input_ids"]
    labels = []
    sep = tokenizer.encode("\nTherapist:")[1:]
    for seq in input_ids:
        # find where the “\nTherapist:” tokens appear
        idx = next((i for i in range(len(seq) - len(sep) + 1)
                    if seq[i : i + len(sep)] == sep), None)
        start = idx + len(sep) if idx is not None else 0
        # mask everything before the reply
        lab = [-100]*start + seq[start:]
        # pad/trunc back to length 256
        labels.append(lab[:256] + [-100]*max(0, 256 - len(lab)))
    tok["labels"] = labels
    return tok

# Process train and test (as validation)
tokenized_train = dataset["train"].map(
    tokenize_function, batched=True, remove_columns=["input","label"]
)
tokenized_val = dataset["test"].map(
    tokenize_function, batched=True, remove_columns=["input","label"]
)

# Training arguments with validation
training_args = TrainingArguments(
    output_dir="./dialoGPT-therapy-lora",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    learning_rate=1e-5,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    fp16=False,
    report_to="none"
)

# Trainer without compute_metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    compute_metrics=None
)

# Train
trainer.train()

# Now do one final eval pass and compute perplexity yourself
metrics = trainer.evaluate(tokenized_val)
print("Final eval loss:", metrics["eval_loss"])
print("Perplexity:", math.exp(metrics["eval_loss"]))

# Save
trainer.save_model("./dialoGPT-therapy-lora")
tokenizer.save_pretrained("./dialogpt-therapy-lora")
