import streamlit as st
from streamlit_audio_recorder import audio_recorder
import speech_recognition as sr
from transformers import pipeline
from pydub import AudioSegment
import plotly.graph_objects as go
import io

# ✅ Must be first Streamlit command
st.set_page_config(page_title="🎙️ Sentiment Analyzer", layout="centered")

# Load sentiment analysis pipeline
analyzer = pipeline("sentiment-analysis")

# Emoji-based suggestions
suggestions = {
    "POSITIVE": "👍 Keep up the positive energy!",
    "NEGATIVE": "🛠 Consider refining your message.",
    "NEUTRAL": "💡 Try to make content more engaging."
}

# Title and option selector
st.title("🎙️ Voice + Text Sentiment Analyzer")
input_mode = st.radio("Choose input method:", ["🎤 Microphone", "⌨️ Manual Text"], horizontal=True)

user_text = ""

if input_mode == "🎤 Microphone":
    st.info("Click 'Start Recording' and then 'Stop Recording' to analyze your voice.")
    audio_bytes = audio_recorder(pause_threshold=1.0)

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        try:
            # Convert to AudioSegment
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)

            # Transcribe using SpeechRecognition
            r = sr.Recognizer()
            with sr.AudioFile(wav_io) as source:
                audio_data = r.record(source)
                user_text = r.recognize_google(audio_data)
                st.success(f"📝 Transcribed Text: {user_text}")

        except Exception as e:
            st.error(f"Audio processing error: {str(e)}")

elif input_mode == "⌨️ Manual Text":
    user_text = st.text_area("Enter your text here:")

# Perform sentiment analysis
if user_text:
    result = analyzer(user_text)[0]
    label = result['label']
    score = result['score']

    st.subheader(f"Sentiment: {label} ({score:.2f})")
    st.write(suggestions.get(label.upper(), "📘"))

    # Plot bar graph
    fig = go.Figure(go.Bar(
        x=[label],
        y=[score],
        marker_color="skyblue",
        name="Confidence"
    ))
    fig.update_layout(title="Sentiment Confidence", yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig)

