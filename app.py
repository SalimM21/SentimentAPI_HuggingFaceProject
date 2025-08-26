import gradio as gr
from transformers import pipeline
import os

# Charger le modèle (identique à celui de ton API)
MODEL_NAME = os.getenv("HF_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
sentiment_pipe = pipeline("sentiment-analysis", model=MODEL_NAME)

# Fonction de prédiction
def predict_sentiment(text):
    result = sentiment_pipe(text)[0]
    label = "positive" if result["label"].upper().startswith("POS") else "negative"
    confidence = float(result["score"])
    return f"Label: {label}, Confidence: {confidence:.4f}"

# Interface Gradio
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Entrez votre texte ici..."),
    outputs="text",
    title="Sentiment Analysis Demo",
    description=f"Modèle: {MODEL_NAME}"
)

# Lancer l'interface en local
if __name__ == "__main__":
    iface.launch(share=True)

