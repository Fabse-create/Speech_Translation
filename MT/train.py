import json
import os
import random
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from datasets import Dataset
import torch
import evaluate
import numpy as np

# Set seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model and tokenizer
MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

# Paths
DATA_DIR = Path("data/downsampled")
OUTPUT_DIR = Path("MT/opus_finetuned")
OUTPUT_DIR.mkdir(exist_ok=True)

# Data loading and preprocessing
def load_data_from_json_files(data_dir):
    """Load all translation pairs from JSON files."""
    translation_pairs = []
    
    json_files = list(Path(data_dir).glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract translation pairs from Files
            if "Files" in data:
                for file_entry in data["Files"]:
                    if "Prompt" in file_entry:
                        prompt = file_entry["Prompt"]
                        if "Transcript" in prompt and "Translation" in prompt:
                            source_text = prompt["Transcript"].strip()
                            target_text = prompt["Translation"].strip()
                            
                            # Only add non-empty pairs
                            if source_text and target_text:
                                translation_pairs.append({
                                    "en": source_text,
                                    "de": target_text
                                })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print(f"Loaded {len(translation_pairs)} translation pairs")
    return translation_pairs

# Load data
translation_pairs = load_data_from_json_files(DATA_DIR)

# Shuffle and split data
random.shuffle(translation_pairs)
total = len(translation_pairs)
train_size = int(0.7 * total)
val_size = int(0.15 * total)

train_data = translation_pairs[:train_size]
val_data = translation_pairs[train_size:train_size + val_size]
test_data = translation_pairs[train_size + val_size:]

print(f"\nData split:")
print(f"Train: {len(train_data)} samples")
print(f"Validation: {len(val_data)} samples")
print(f"Test: {len(test_data)} samples")

# Save split data
with open(OUTPUT_DIR / "train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / "validation.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / "test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(f"\nData splits saved to {OUTPUT_DIR}/")

# Preprocessing function for tokenization
def preprocess_function(examples):
    """Tokenize source and target texts in batches."""
    model_inputs = tokenizer(
        examples["en"],
        max_length=128,
        truncation=True,
        # Remove padding="max_length" - let DataCollator handle it
    )
    
    # Use text_target instead of deprecated as_target_tokenizer()
    labels = tokenizer(
        text_target=examples["de"],
        max_length=128,
        truncation=True,
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Create datasets
train_dataset = Dataset.from_dict({
    "en": [d["en"] for d in train_data],
    "de": [d["de"] for d in train_data]
})

val_dataset = Dataset.from_dict({
    "en": [d["en"] for d in val_data],
    "de": [d["de"] for d in val_data]
})

test_dataset = Dataset.from_dict({
    "en": [d["en"] for d in test_data],
    "de": [d["de"] for d in test_data]
})

# Tokenize datasets
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,  # <-- Changed from False
    remove_columns=["en", "de"]
)

val_dataset = val_dataset.map(
    preprocess_function,
    batched=True,  # <-- Changed from False
    remove_columns=["en", "de"]
)

test_dataset = test_dataset.map(
    preprocess_function,
    batched=True,  # <-- Changed from False
    remove_columns=["en", "de"]
)


print(f"\nDatasets prepared for training")

# Load BLEU metric
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    """Compute BLEU score for predictions."""
    preds, labels = eval_preds
    
    # In case the model returns more than just the predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Replace -100s in labels (used for padding in loss calculation)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # SacreBLEU expects references as list of lists
    decoded_labels = [[label] for label in decoded_labels]
    
    # Compute BLEU score
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    # Also compute average prediction length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    
    return {
        "bleu": round(result["score"], 2),
        "gen_len": round(np.mean(prediction_lens), 1)
    }

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=OUTPUT_DIR / "logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    report_to="none", 
    predict_with_generate=True,  # <-- ADD THIS LINE
    metric_for_best_model="bleu",  # <-- ADD THIS LINE
    greater_is_better=True,  # <-- ADD THIS LINE
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics, 
)

print("\nStarting finetuning...")
trainer.train()

# Save the finetuned model
model.save_pretrained(OUTPUT_DIR / "best_model")
tokenizer.save_pretrained(OUTPUT_DIR / "best_model")
print(f"Model saved to {OUTPUT_DIR}/best_model")

# Evaluate on test set
print("\nEvaluating on test set...")
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test results: {test_results}")

# Save test results
with open(OUTPUT_DIR / "test_results.json", "w") as f:
    json.dump(test_results, f, indent=2)

print("\nFinetuning completed!")
