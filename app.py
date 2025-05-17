import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import pipeline

# Load model and tokenizer
@st.cache_resource
def load_pipeline():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiment_pipeline = load_pipeline()

# Streamlit UI
st.title("Sentiment Analysis with RoBERTa")

user_input = st.text_area("Enter text to analyze sentiment", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            results = sentiment_pipeline(user_input)
            label = results[0]['label']
            score = results[0]['score']
            st.success(f"**Sentiment:** {label} (Confidence: {score:.2f})")
