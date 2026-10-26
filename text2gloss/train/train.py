import os
from pathlib import Path
import numpy as np
import torch
from jiwer import wer
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# Relative paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DATA_FILE = REPO_ROOT / "data" / "Corpus-Vie-VSL-10k" / "Corpus-Vie-VSL-10K.csv"
OUTPUT_DIR = REPO_ROOT / "models" / "vit5_sign_grammar"
MODEL_NAME_OR_PATH = os.getenv("BASE_MODEL_NAME", "VietAI/vit5-base")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Remove pad tokens
    refs = [r.replace(tokenizer.pad_token, "").strip() for r in refs]

    wers = [wer(r, p) for r, p in zip(refs, preds)]
    return {"wer": np.mean(wers)}


print(f"Loading pretrained model and tokenizer: {MODEL_NAME_OR_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)

print(f"Loading parallel corpus from: {DATA_FILE}...")
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Training dataset not found at: {DATA_FILE}")

dataset = load_dataset("csv", data_files={"train": str(DATA_FILE)})


def preprocess_function(examples):
    inputs = ["sign-grammar: " + ex for ex in examples["input"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(text_target=examples["output"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = dataset.map(preprocess_function, batched=True)
split_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=10,
    predict_with_generate=True,
    logging_dir=str(OUTPUT_DIR / "logs"),
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

if __name__ == "__main__":
    print("Starting training...")
    trainer.train()
    print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Training finished successfully!")
