import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
import speech_recognition as sr
from langdetect import detect
import tempfile
import torch
import matplotlib.pyplot as plt

st.set_page_config(page_title="🎙️ Sentiment Analyzer", layout="centered")

@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

classifier = load_model()

label_map = {
    "LABEL_0": ("Negative", "😠"),
    "LABEL_1": ("Neutral", "😐"),
    "LABEL_2": ("Positive", "😊")
}

suggestions = {
    "Negative": "❗ Address negative feedback with improved service or product changes.",
    "Neutral": "💡 Try to make content more engaging.",
    "Positive": "✅ Keep up the great work and promote customer reviews!"
}

st.title("🔍 RoBERTa Sentiment Analyzer with Text & Audio Upload Support")
st.markdown("Analyze **text** or **upload audio** for sentiment.")

# 1. Text Input
st.header("📝 Enter Text")
text_input = st.text_area("Type or paste text below:", placeholder="e.g., I love the product quality!", height=150)

# 2. Audio Upload
st.header("📁 Upload Audio File")
audio_file = st.file_uploader("Upload WAV or MP3 file", type=["wav", "mp3"])

transcribed_text = ""

if audio_file:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)

        transcribed_text = recognizer.recognize_google(audio_data)
        st.success(f"🎧 Transcribed from upload: {transcribed_text}")
    except Exception as e:
        st.error(f"❌ Audio processing error: {e}")

# Choose which text to analyze
final_text = transcribed_text if transcribed_text else text_input

# Sentiment Analysis Button
if st.button("🔍 Analyze Sentiment"):
    if final_text.strip():
        with st.spinner("Analyzing..."):
            try:
                lang = detect(final_text)
                st.info(f"🌐 Detected Language: {lang.upper()}")
            except:
                st.info("🌐 Language detection failed")

            result = classifier(final_text)[0]
            label_code = result['label']
            label, emoji = label_map.get(label_code, ("Unknown", "❓"))
            confidence = result['score'] * 100

            st.success(f"Sentiment: **{label}** {emoji}")
            st.write(f"Confidence: **{confidence:.2f}%**")
            st.info(suggestions.get(label, ""))

            # Visualization
            st.subheader("📊 Sentiment Confidence")
            labels = ["Negative", "Neutral", "Positive"]
            scores = [0, 0, 0]
            try:
                index = int(label_code[-1])
                scores[index] = confidence / 100.0  # fraction
            except:
                pass

            fig, ax = plt.subplots()
            ax.bar(labels, scores, color=["red", "gray", "green"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Confidence")
            st.pyplot(fig)
    else:
        st.warning("⚠️ Please enter text or upload an audio file.")

st.markdown("<hr><center>Made with 🤖 RoBERTa & 🎙️ Streamlit</center>", unsafe_allow_html=True)



