import streamlit as st
from streamlit_audio_recorder import audio_recorder
import speech_recognition as sr
from transformers import pipeline
from pydub import AudioSegment
import io

# Set config FIRST
st.set_page_config(page_title="🎙️ Sentiment Analyzer", layout="centered")

# Title
st.title("🎙️ Voice & Text Sentiment Analyzer")

# Sentiment pipeline (HuggingFace)
sentiment_analyzer = pipeline("sentiment-analysis")

# --- Voice Input ---
st.header("🔴 Record Your Voice")
audio_bytes = audio_recorder(
    text="Click to record",
    recording_color="#e53935",
    neutral_color="#6c757d",
    icon_name="mic"
)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    st.info("Transcribing audio...")

    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            transcribed_text = recognizer.recognize_google(audio_data)
            st.success(f"Transcribed: {transcribed_text}")

            sentiment = sentiment_analyzer(transcribed_text)[0]
            st.subheader("📊 Sentiment Result")
            st.write(f"**Label:** {sentiment['label']}")
            st.write(f"**Confidence:** {sentiment['score']:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")

# --- Text Input ---
st.header("✍️ Or Enter Text Manually")
text_input = st.text_area("Enter your sentence here:")

if st.button("Analyze Sentiment") and text_input:
    result = sentiment_analyzer(text_input)[0]
    st.subheader("📊 Sentiment Result")
    st.write(f"**Label:** {result['label']}")
    st.write(f"**Confidence:** {result['score']:.2f}")


