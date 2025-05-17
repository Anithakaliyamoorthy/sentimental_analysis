import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import pipeline

@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

st.title("Sentiment Analysis App")

text = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze"):
    if text.strip():
        with st.spinner("Analyzing..."):
            classifier = load_model()
            result = classifier(text)[0]
            st.success(f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})")
    else:
        st.warning("Please enter some text.")
