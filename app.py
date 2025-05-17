import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load tokenizer and model
model_path = "./roberta_sentiment_model"  # Adjust path if needed
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[predicted_class]

# Streamlit app
st.set_page_config(page_title="Sentiment Classifier", page_icon="🔍")
st.title("📝 Amazon Review Sentiment Classifier")
st.write("Enter a product review below and see its predicted sentiment.")

review_text = st.text_area("Review Text", height=150)

if st.button("Analyze"):
    if review_text.strip():
        sentiment = predict_sentiment(review_text)
        st.success(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter some text.")
