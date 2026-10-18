"""
predict.py
Single-text prediction using the trained model.
Owner: Zaka
"""

import torch

try:
    from .model import EmotionClassifier
    from .preprocessing import encode_text
    from .config import EMOTION_LABELS, N_CLASSES, MODEL_SAVE_PATH
except ImportError:
    from model import EmotionClassifier
    from preprocessing import encode_text
    from config import EMOTION_LABELS, N_CLASSES, MODEL_SAVE_PATH


def load_model(device):
    model = EmotionClassifier(n_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    return model


def predict_emotion(text, model, device):
    inputs = encode_text(text)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)

    predicted_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_idx].item()

    return EMOTION_LABELS[predicted_idx], confidence, probs.cpu().numpy()[0]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    sample = "I can't believe this happened!! 😱"
    emotion, confidence, all_probs = predict_emotion(sample, model, device)
    print(f"Text: {sample}")
    print(f"Predicted: {emotion} ({confidence:.2%})")