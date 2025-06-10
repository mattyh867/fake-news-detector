from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch

dataset = load_dataset("csv", data_files={"train": "data/train.csv"}, split="train")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = RobertaForSequenceClassification.from_pretrained("roberta-base")

training_args = TrainingArguments(
    output_dir="./roberta-fake-news",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_total_limit=1,
    evaluation_strategy="no",
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

model.save_pretrained("../backend/model")
tokenizer.save_pretrained("../backend/model/tokenizer")