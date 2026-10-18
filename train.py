"""
train.py
Training loop with class weighting and learning rate decay.
Owner: Mohammed
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from model import EmotionClassifier
from preprocessing import build_dataloaders, compute_class_weights_tensor
from config import N_CLASSES, LEARNING_RATE, N_EPOCHS, MODEL_SAVE_PATH



def train_model(train_loader, val_loader, train_labels, device,
                 n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE, use_class_weights=True):

    model = EmotionClassifier(n_classes=N_CLASSES).to(device)

    if use_class_weights:
        class_weights = compute_class_weights_tensor(train_labels, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    total_steps = len(train_loader) * n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate_during_training(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{n_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model


def evaluate_during_training(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(val_loader), correct / total


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Créer le dossier outputs/ automatiquement s'il n'existe pas
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    train_loader, val_loader, test_loader, train_labels = build_dataloaders()
    model = train_model(train_loader, val_loader, train_labels, device)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")