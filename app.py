import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Cache model
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Map model labels to sentiment and emoji
label_map = {
    "LABEL_0": ("Negative", "😞"),
    "LABEL_1": ("Neutral", "😐"),
    "LABEL_2": ("Positive", "😊")
}

# Track sentiment history
if "history" not in st.session_state:
    st.session_state.history = []

# Page settings
st.set_page_config(page_title="💬 Sentiment Analyzer", layout="centered")
st.title("💬 Sentiment Analysis App")

st.markdown("Enter your text and click **Analyze** to get sentiment with confidence.")

# Text input
text = st.text_area("Enter text for sentiment analysis:")

col1, col2 = st.columns([1, 1])
analyze = col1.button("🔍 Analyze")
clear = col2.button("🧹 Clear")

# Handle clear
if clear:
    st.experimental_rerun()

if analyze and text.strip():
    with st.spinner("Analyzing..."):
        classifier = load_model()
        result = classifier(text)[0]

        label = result["label"]
        score = result["score"]

        sentiment, emoji = label_map.get(label, ("Unknown", "❓"))

        # Save to history
        st.session_state.history.append((text, sentiment, score))

        # Output
        st.success(f"**Sentiment:** {sentiment} {emoji}")
        st.progress(score)
        st.info(f"**Confidence:** {score:.2%}")

        # Word cloud
        wc = WordCloud(width=500, height=200, background_color='white').generate(text)
        st.subheader("🧠 Word Cloud of Your Text")
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

elif analyze:
    st.warning("⚠️ Please enter some text.")

# Show sentiment history
if st.session_state.history:
    st.sidebar.header("🕒 Sentiment History")
    for i, (txt, sent, sc) in enumerate(reversed(st.session_state.history[-5:])):
        st.sidebar.markdown(f"**{sent}** ({sc:.0%})")
        st.sidebar.caption(f"`{txt[:50]}...`")

