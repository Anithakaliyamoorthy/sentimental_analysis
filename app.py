import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import pipeline

# Load model and tokenizer only once (cached)
@st.cache_resource
def load_sentiment_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_pipeline

# Streamlit app UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("💬 Twitter-RoBERTa Sentiment Analyzer")

st.markdown("Enter some text and click **Analyze** to see if it's Positive, Neutral, or Negative.")

# Input text box
user_input = st.text_area("Enter your text below:", height=150)

# On click Analyze
if st.button("Analyze"):
    if user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            classifier = load_sentiment_model()
            result = classifier(user_input)[0]
            label = result["label"]
            confidence = result["score"]

            # Label formatting
            if label == "LABEL_0":
                sentiment = "Negative"
                emoji = "😞"
            elif label == "LABEL_1":
                sentiment = "Neutral"
                emoji = "😐"
            elif label == "LABEL_2":
                sentiment = "Positive"
                emoji = "😊"
            else:
                sentiment = "Unknown"
                emoji = "❓"

            st.success(f"**Sentiment:** {sentiment} {emoji}")
            st.info(f"**Confidence:** {confidence:.2%}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
