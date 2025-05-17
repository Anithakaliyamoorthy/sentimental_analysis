import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load tokenizer and model
model_path = "roberta_sentiment_model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

# Set model to evaluation mode and move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[prediction]

# Streamlit UI
st.title("📝 Amazon Review Sentiment Classifier")
st.write("Enter a product review and see the predicted sentiment!")

user_input = st.text_area("Enter Review Text", height=150)

if st.button("Predict"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter some text to analyze.")
