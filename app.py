import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

classifier = load_model()

# Emoji mapping
label_map = {
    "LABEL_0": ("Negative", "😠"),
    "LABEL_1": ("Neutral", "😐"),
    "LABEL_2": ("Positive", "😊")
}

# Sentiment improvement suggestions
improvement_suggestion = {
    "Negative": "❗ Consider improving product quality or addressing key user complaints.",
    "Neutral": "💡 Users feel neutral — enhancing value or adding features might improve engagement.",
    "Positive": "✅ Users are happy — maintain the quality and consider gathering testimonials!"
}

# App layout
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("💬 Sentiment Analysis App")
st.markdown("Enter your text and click **Analyze** to get sentiment with confidence.")

# Text input
text = st.text_area("Enter text for sentiment analysis:", height=150, placeholder="e.g., The product quality is average, not too good or bad.")

# Analyze button
if st.button("Analyze"):
    if text.strip():
        with st.spinner("Analyzing..."):
            result = classifier(text)[0]
            label_code = result['label']
            label, emoji = label_map[label_code]
            score = result['score'] * 100

            # Display sentiment
            st.markdown(f"""
            <div style='background-color:#e8f5e9;padding:1.2rem;border-radius:10px;'>
                <b>Sentiment:</b> {label} {emoji}
            </div>
            """, unsafe_allow_html=True)

            # Display confidence
            st.markdown(f"""
            <div style='background-color:#e3f2fd;padding:1rem;border-radius:10px;'>
                <b>Confidence:</b> {score:.2f}%
            </div>
            """, unsafe_allow_html=True)

            # Display improvement suggestion
            st.info(improvement_suggestion[label])

            
    else:
        st.warning("Please enter some text before analyzing.")

# Footer style
st.markdown("<br><hr style='border:0.5px solid #ddd;'><br>", unsafe_allow_html=True)

