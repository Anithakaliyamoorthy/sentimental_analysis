import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import pipeline

# Load model and tokenizer only once
@st.cache_resource
def load_sentiment_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_pipeline

# Label mapping for RoBERTa model
label_map = {
    "LABEL_0": ("Negative", "😞"),
    "LABEL_1": ("Neutral", "😐"),
    "LABEL_2": ("Positive", "😊")
}

# Streamlit app UI
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("💬 Sentiment Analysis App")

st.markdown("Enter your text and click **Analyze** to get sentiment with confidence.")

# Text input
user_input = st.text_area("Enter text for sentiment analysis:")

# On click Analyze
if st.button("Analyze"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            classifier = load_sentiment_model()
            result = classifier(user_input)[0]

            label = result["label"]
            score = result["score"]

            sentiment, emoji = label_map.get(label, ("Unknown", "❓"))

            st.success(f"**Sentiment:** {sentiment} {emoji}")
            st.info(f"**Confidence:** {score:.2%}")
    else:
        st.warning("⚠️ Please enter some text.")
