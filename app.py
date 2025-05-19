import streamlit as st
import speech_recognition as sr
import tempfile
import os
from transformers import pipeline
import plotly.express as px

# 🔧 Set Streamlit page config FIRST
st.set_page_config(page_title="🎙️ Sentiment Voice Analyzer", layout="centered")

# 🎯 Load sentiment model (DistilBERT) - cached for efficiency
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_analyzer = load_model()

# 📌 Label map with emojis
label_map = {
    "LABEL_0": ("Negative", "😠"),
    "LABEL_1": ("Neutral", "😐"),
    "LABEL_2": ("Positive", "😊")
}

# UI Title
st.title("🎙️ Sentiment Analyzer from Voice & Text")

# Tabs for Audio or Text Input
tab1, tab2 = st.tabs(["🎧 Audio Upload", "📝 Text Input"])

with tab1:
    st.write("Upload an audio file (WAV, MP3, FLAC, AIFF) for transcription and sentiment analysis.")
    uploaded_file = st.file_uploader("🔊 Upload Audio", type=["wav", "mp3", "flac", "aiff"])

    if uploaded_file:
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(tmp_path) as source:
                audio = recognizer.record(source)
            transcribed_text = recognizer.recognize_google(audio)
            st.success("✅ Transcription successful!")
            st.markdown(f"**📝 Transcribed Text:** {transcribed_text}")

            # Sentiment analysis
            result = sentiment_analyzer(transcribed_text)[0]
            label = result["label"]
            score = round(result["score"] * 100, 2)

            # Map label to emoji and name
            sentiment_name, emoji = label_map.get(label, ("Unknown", "❓"))

            st.subheader(f"{emoji} {sentiment_name} ({score}%)")

            # Visualization with Plotly
            fig = px.bar(
                x=["Confidence"],
                y=[score],
                labels={"x": "Sentiment Confidence (%)", "y": "Score"},
                title=f"Sentiment Confidence: {sentiment_name}",
                range_y=[0, 100],
            )
            st.plotly_chart(fig)

        except sr.UnknownValueError:
            st.error("😕 Could not understand audio.")
        except sr.RequestError as e:
            st.error(f"🔌 Google Speech Recognition error: {e}")
        except Exception as e:
            st.error(f"⚠️ Audio processing error: {e}")

        os.remove(tmp_path)

with tab2:
    st.write("Or directly enter text below to analyze sentiment.")
    user_text = st.text_area("Enter text for sentiment analysis", height=150)

    if st.button("Analyze Sentiment", key="text_analyze") and user_text.strip() != "":
        result = sentiment_analyzer(user_text)[0]
        label = result["label"]
        score = round(result["score"] * 100, 2)
        sentiment_name, emoji = label_map.get(label, ("Unknown", "❓"))

        st.subheader(f"{emoji} {sentiment_name} ({score}%)")

        fig = px.bar(
            x=["Confidence"],
            y=[score],
            labels={"x": "Sentiment Confidence (%)", "y": "Score"},
            title=f"Sentiment Confidence: {sentiment_name}",
            range_y=[0, 100],
        )
        st.plotly_chart(fig)



