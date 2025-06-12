import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import gc

# Check GPU availability
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("No GPU available, using CPU")
    device = torch.device("cpu")

# Enable aggressive garbage collection
gc.enable()

# Function to clear GPU memory if available
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Clear memory at start
clear_memory()

# Read and combine the datasets
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Add labels (1 for fake, 0 for true)
fake_df['label'] = 1
true_df['label'] = 0

# Combine the datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Combine title and text for better prediction
df['full_text'] = df['title'] + " " + df['text']

# Create dataset
dataset = Dataset.from_pandas(df[['full_text', 'label']])

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(batch):
    return tokenizer(batch["full_text"], padding=True, truncation=True, max_length=512)

# Prepare dataset
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Split into train and validation sets
dataset = dataset.train_test_split(test_size=0.1)

# Load and prepare the model
model = RobertaForSequenceClassification.from_pretrained("roberta-base").to(device)
model.gradient_checkpointing_enable()  # Enable gradient checkpointing

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./roberta-fake-news",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Increased from 2 to 4
    per_device_eval_batch_size=4,  # Increased from 2 to 4
    save_total_limit=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    gradient_accumulation_steps=2,  # Reduced from 4 to 2 to maintain same effective batch size
    gradient_checkpointing=True,
    optim="adafactor",
    dataloader_num_workers=0,
    group_by_length=True
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# Train the model
trainer.train()

# Clear memory before saving
clear_memory()

# Save the model and tokenizer
model.save_pretrained("../backend/model")
tokenizer.save_pretrained("../backend/model/tokenizer")

# After training completes
model.push_to_hub("mattyh867/fake-news-detector")
tokenizer.push_to_hub("mattyh867/fake-news-detector")

# Final memory cleanup
clear_memory()