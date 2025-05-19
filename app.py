import streamlit as st
import speech_recognition as sr
import tempfile
import os
import torch
from transformers import pipeline
import plotly.express as px

# 🔧 Set Streamlit page config FIRST
st.set_page_config(page_title="🎙️ Sentiment Voice Analyzer", layout="centered")

# 🎯 Load sentiment model (DistilBERT)
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_analyzer = load_model()

# 🎨 Sentiment icons & suggestions
sentiment_styles = {
    "POSITIVE": ("🟢 Positive", "👍 Keep it up!"),
    "NEGATIVE": ("🔴 Negative", "⚠️ Try to improve tone."),
    "NEUTRAL": ("🟡 Neutral", "💡 Try to make it more engaging.")
}

# 📌 App UI
st.title("🎙️ Sentiment Analyzer from Voice")
st.write("Upload an audio file (WAV recommended), and we'll transcribe it and analyze sentiment.")

uploaded_file = st.file_uploader("🔊 Upload Audio", type=["wav", "mp3", "flac", "aiff"])

if uploaded_file:
    # Save audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # 🗣️ Transcribe using Google Speech Recognition
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)
        transcribed_text = recognizer.recognize_google(audio)
        st.success("✅ Transcription successful!")
        st.markdown(f"**📝 Transcribed Text:** {transcribed_text}")
        
        # ✨ Sentiment Analysis
        sentiment_result = sentiment_analyzer(transcribed_text)[0]
        label = sentiment_result["label"].upper()
        score = round(sentiment_result["score"] * 100, 2)

        # Display result
        icon, suggestion = sentiment_styles.get(label, ("⚪ Unknown", "No advice"))
        st.subheader(f"{icon} ({score}%)")
        st.info(suggestion)

        # 📊 Plotly Visualization
        fig = px.bar(
            x=["Positive", "Negative"],
            y=[sentiment_result["score"] if label == "POSITIVE" else 1 - sentiment_result["score"],
               sentiment_result["score"] if label == "NEGATIVE" else 1 - sentiment_result["score"]],
            labels={"x": "Sentiment", "y": "Confidence"},
            color=["Positive", "Negative"],
            color_discrete_map={"Positive": "green", "Negative": "red"},
            title="Sentiment Confidence"
        )
        st.plotly_chart(fig)

    except sr.UnknownValueError:
        st.error("😕 Google Speech Recognition couldn't understand the audio.")
    except sr.RequestError as e:
        st.error(f"🔌 Could not request results from Google Speech Recognition service: {e}")
    except Exception as e:
        st.error(f"⚠️ Audio processing error: {e}")

    os.remove(tmp_path)

