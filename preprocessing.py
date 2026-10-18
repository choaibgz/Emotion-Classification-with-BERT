"""
preprocessing.py
Text cleaning (without destroying emotional signal), tokenization,
and class balancing utilities.
Owner: Choaib
"""

import re
import emoji
import numpy as np
import torch
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from config import DATASET_NAME, MODEL_NAME, MAX_LENGTH, BATCH_SIZE

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


# ---------------------------------------------------------
# 1. Text cleaning WITHOUT losing emotional information
# ---------------------------------------------------------

def clean_text(text):
    """
    Cleaning adapted to BERT fine-tuning:
    - removes only real noise (URLs, mentions)
    - keeps punctuation (!, ?) since it carries emotional intensity
    - keeps stopwords/natural sentence structure (needed by BERT)
    - converts emojis to descriptive text instead of deleting them
    """
    text = text.strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = emoji.demojize(text, delimiters=(" :", ": "))
    return text.strip()


def clean_dataset_batch(examples):
    examples["text"] = [clean_text(t) for t in examples["text"]]
    return examples


# ---------------------------------------------------------
# 2. Tokenization
# ---------------------------------------------------------

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )


def encode_text(text):
    """Used at inference time in predict.py / app.py."""
    cleaned = clean_text(text)
    return tokenizer(
        cleaned,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )


# ---------------------------------------------------------
# 3. Full pipeline: load, clean, tokenize
# ---------------------------------------------------------

def build_dataloaders(batch_size=BATCH_SIZE):
    raw_dataset = load_dataset(DATASET_NAME)

    cleaned_dataset = raw_dataset.map(clean_dataset_batch, batched=True)
    tokenized_datasets = cleaned_dataset.map(tokenize_function, batched=True)

    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    train_loader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size)
    test_loader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)

    train_labels = np.array(raw_dataset["train"]["label"])

    return train_loader, val_loader, test_loader, train_labels


# ---------------------------------------------------------
# 4. Class weights (handles 'surprise' imbalance)
# ---------------------------------------------------------

def compute_class_weights_tensor(labels, device):
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float32).to(device)


# ---------------------------------------------------------
# 5. SMOTE (alternative, on embeddings only — not raw text)
# ---------------------------------------------------------

def apply_smote_on_embeddings(embeddings, labels, k_neighbors=5):
    from imblearn.over_sampling import SMOTE
    print("Distribution before SMOTE:", np.bincount(labels))
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    embeddings_res, labels_res = smote.fit_resample(embeddings, labels)
    print("Distribution after SMOTE:", np.bincount(labels_res))
    return embeddings_res, labels_res


if __name__ == "__main__":
    train_loader, val_loader, test_loader, train_labels = build_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    sample = "I can't believe this happened!! 😱"
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))