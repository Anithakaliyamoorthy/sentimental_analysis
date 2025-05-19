import streamlit as st
from transformers import pipeline
from langdetect import detect
from googletrans import Translator
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import os

# Set page configuration
st.set_page_config(page_title="RoBERTa Sentiment Analyzer", layout="centered")

# Load RoBERTa model
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    return classifier

classifier = load_model()
translator = Translator()

# Label map for emoji and suggestion
label_map = {
    "LABEL_0": ("Negative", "😠"),
    "LABEL_1": ("Neutral", "😐"),
    "LABEL_2": ("Positive", "😊")
}

suggestions = {
    "Negative": "❗ Address negative feedback with improved service or product changes.",
    "Neutral": "💡 Consider improving engagement with more interactive or valuable content.",
    "Positive": "✅ Maintain quality and encourage customer reviews to build trust!"
}

# UI Layout
st.title("🔍 Sentiment Analyzer with Audio & Translation")
st.markdown("Analyze sentiment from **text** or **audio** input. Supports multilingual input and audio transcription.")

# Input: Text
st.header("📝 Enter Text")
text_input = st.text_area("Type or paste text below:", placeholder="I love the product quality!", height=150)

# Input: Audio
st.header("🎤 Upload Audio (MP3 or WAV)")
audio_file = st.file_uploader("Upload audio file for sentiment analysis", type=["mp3", "wav"])

def convert_audio(file, format):
    sound = AudioSegment.from_file(file)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sound.export(temp_file.name, format="wav")
    return temp_file.name

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

transcribed_text = ""
if audio_file:
    try:
        audio_format = audio_file.name.split('.')[-1]
        if audio_format not in ["wav", "mp3"]:
            st.error("Unsupported audio format.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}") as tmp:
                tmp.write(audio_file.read())
                audio_path = convert_audio(tmp.name, "wav") if audio_format == "mp3" else tmp.name
                transcribed_text = transcribe_audio(audio_path)
                st.success(f"Transcribed Text: {transcribed_text}")
    except Exception as e:
        st.error(f"Audio processing error: {e}")

# Final text to analyze
text_to_analyze = transcribed_text if transcribed_text else text_input

# Analyze Button
if st.button("🔍 Analyze Sentiment"):
    if text_to_analyze.strip():
        with st.spinner("Analyzing..."):
            lang = detect(text_to_analyze)
            st.info(f"🌐 Detected Language: {lang.upper()}")

            if lang != "en":
                try:
                    translated = translator.translate(text_to_analyze, src=lang, dest="en").text
                    st.markdown(f"🗨️ Translated Text (EN): *{translated}*")
                    text_to_analyze = translated
                except Exception as e:
                    st.warning(f"Translation failed: {e}")

            result = classifier(text_to_analyze)[0]
            label_code = result['label']
            label, emoji = label_map[label_code]
            score = result['score'] * 100

            st.success(f"Sentiment: **{label}** {emoji}")
            st.write(f"Confidence Score: **{score:.2f}%**")
            st.info(suggestions[label])
    else:
        st.warning("Please enter text or upload an audio file to analyze.")

# Footer
st.markdown("<hr><center>Built with 🤖 RoBERTa, Streamlit, and Google Translate</center>", unsafe_allow_html=True)
