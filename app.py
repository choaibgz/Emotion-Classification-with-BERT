"""
app.py
Streamlit interface for emotion prediction.
Owner: Zaka
"""

import streamlit as st
import torch

try:
    from .predict import load_model, predict_emotion
    from .config import EMOTION_LABELS
except ImportError:
    from predict import load_model, predict_emotion
    from config import EMOTION_LABELS

st.set_page_config(page_title="Emotion Detector", page_icon="🎭")

st.title("🎭 Détecteur d'émotions")
st.write("Analysez l'émotion exprimée dans un texte (BERT fine-tuné).")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def get_model():
    return load_model(device)


model = get_model()

text_input = st.text_area("Entrez votre texte :", placeholder="I can't believe this!! 😱")

if st.button("Analyser"):
    if text_input.strip():
        emotion, confidence, all_probs = predict_emotion(text_input, model, device)

        st.subheader(f"Émotion détectée : **{emotion}**")
        st.write(f"Confiance : {confidence:.2%}")

        st.bar_chart(dict(zip(EMOTION_LABELS, all_probs)))
    else:
        st.warning("Merci d'entrer un texte.")