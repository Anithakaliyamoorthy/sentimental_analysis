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

# App layout
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("💬 Sentiment Analysis App")
st.markdown("Enter your text and click **Analyze** to get sentiment with confidence.")

# App features description
st.markdown("""
### 🔍 App Features:
- Analyze sentiment as **Positive**, **Neutral**, or **Negative**.
- Provides confidence score and **emoji-based feedback**.
- Generates a **word cloud** from your input.
- Keeps a history of the last few analyses.
- Works with real product reviews or your own texts.

Just enter any sentence (like an Amazon product review), and hit **Analyze**!
""")

# Text input
text = st.text_area("Enter text for sentiment analysis:", height=150, placeholder="e.g., The product quality is average, not too good or bad.")

# Example buttons
example_col1, example_col2, example_col3 = st.columns(3)
if example_col1.button("Example Positive"):
    text = "I absolutely love this product! Works great."
if example_col2.button("Example Neutral"):
    text = "The product is okay. Does the job."
if example_col3.button("Example Negative"):
    text = "Terrible experience, completely disappointed."

# Analyze button
if st.button("Analyze"):
    if text.strip():
        with st.spinner("Analyzing..."):
            result = classifier(text)[0]
            label, emoji = label_map[result['label']]
            score = result['score'] * 100

            st.markdown(f"""
            <div style='background-color:#e8f5e9;padding:1.2rem;border-radius:10px;'>
                <b>Sentiment:</b> {label} {emoji}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background-color:#e3f2fd;padding:1rem;border-radius:10px;'>
                <b>Confidence:</b> {score:.2f}%
            </div>
            """, unsafe_allow_html=True)

            # Word cloud
            st.subheader("Word Cloud of Input Text")
            wordcloud = WordCloud(background_color='white').generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
    else:
        st.warning("Please enter some text before analyzing.")

# Styling with some padding at the bottom
st.markdown("<br><hr style='border:0.5px solid #ddd;'><br>", unsafe_allow_html=True)
