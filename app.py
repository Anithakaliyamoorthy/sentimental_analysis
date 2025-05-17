import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import matplotlib.pyplot as plt
from wordcloud import WordCloud

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

LABELS = {
    0: ("Negative", "😞"),
    1: ("Neutral", "😐"),
    2: ("Positive", "😊")
}

@st.cache_resource
def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

st.set_page_config(page_title="💬 Sentiment Analyzer", layout="centered")
st.title("💬 Sentiment Analysis App")
st.markdown("Enter your text and click **Analyze** to get sentiment and explanation.")

text = st.text_area("📝 Enter text for sentiment analysis:")

col1, col2 = st.columns([1, 1])
analyze = col1.button("🔍 Analyze")
clear = col2.button("🧹 Clear")

if clear:
    st.experimental_rerun()

if "history" not in st.session_state:
    st.session_state.history = []

if analyze and text.strip():
    with st.spinner("Analyzing..."):
        classifier = load_pipeline()
        result = classifier(text)[0]

        # Extract label index from result
        label_id = int(result['label'].split("_")[-1])
        sentiment, emoji, explanation = LABELS[label_id]
        confidence = result['score']

        st.session_state.history.append((text, sentiment, confidence))

        # Display results
        st.success(f"**Sentiment:** {sentiment} {emoji}")
        st.info(f"**Confidence:** {confidence:.2%}")
        st.write(f"**Why?** {explanation}")

        # Word cloud
        st.subheader("☁️ Word Cloud")
        wordcloud = WordCloud(width=500, height=200, background_color='white').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

elif analyze:
    st.warning("⚠️ Please enter some text to analyze.")

# Sidebar: History
if st.session_state.history:
    st.sidebar.header("🕒 Sentiment History")
    for i, (txt, sent, score) in enumerate(reversed(st.session_state.history[-5:])):
        st.sidebar.markdown(f"**{sent}** ({score:.0%})")
        st.sidebar.caption(f"`{txt[:50]}...`")

