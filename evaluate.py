"""
evaluate.py
Final evaluation on the untouched test set.
Owner: Oumkeltoum
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import EmotionClassifier
from preprocessing import build_dataloaders
from config import EMOTION_LABELS, N_CLASSES, MODEL_SAVE_PATH, CONFUSION_MATRIX_PATH


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def print_metrics(y_true, y_pred):
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print(f"F1-macro:    {f1_macro:.4f}")
    print(f"F1-weighted: {f1_weighted:.4f}")
    print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS))

    return f1_macro, f1_weighted


def plot_confusion_matrix(y_true, y_pred, save_path=CONFUSION_MATRIX_PATH):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Emotion Classification")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, _ = build_dataloaders()

    model = EmotionClassifier(n_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    y_true, y_pred = evaluate_model(model, test_loader, device)
    print_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)