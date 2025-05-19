import streamlit as st
import speech_recognition as sr
import tempfile
import os
from transformers import pipeline
import plotly.express as px

# 🔧 Streamlit config - Must be first
st.set_page_config(page_title="🎙️ Sentiment Voice & Text Analyzer", layout="centered")

# 🎯 Load sentiment analysis model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_analyzer = load_model()

# 💡 Style definitions
sentiment_styles = {
    "POSITIVE": ("🟢 Positive", "👍 Keep it up!"),
    "NEGATIVE": ("🔴 Negative", "⚠️ Try to improve tone."),
    "NEUTRAL": ("🟡 Neutral", "💡 Try to make content more engaging.")
}

st.title("🎙️📄 Sentiment Analyzer")
st.write("You can either upload a voice file (WAV recommended) or directly type your text for sentiment analysis.")

# 📥 Audio upload
uploaded_file = st.file_uploader("🔊 Upload Audio File", type=["wav", "mp3", "flac", "aiff"])

# 🧾 Text input
text_input = st.text_area("📝 Or type your text here:", placeholder="Type your message here...")

def analyze_sentiment(text):
    sentiment_result = sentiment_analyzer(text)[0]
    label = sentiment_result["label"].upper()
    score = round(sentiment_result["score"] * 100, 2)
    icon, suggestion = sentiment_styles.get(label, ("⚪ Unknown", "No advice available."))

    # 💬 Display Results
    st.subheader(f"{icon} ({score}%)")
    st.info(suggestion)

    # 📊 Plot confidence
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

# 🔎 Process text input
if text_input:
    st.write("Analyzing text sentiment...")
    analyze_sentiment(text_input)

# 🎤 Process audio input
elif uploaded_file:
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
        analyze_sentiment(transcribed_text)

    except sr.UnknownValueError:
        st.error("😕 Could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"🔌 Could not reach Google API: {e}")
    except Exception as e:
        st.error(f"⚠️ Audio error: {e}")
    finally:
        os.remove(tmp_path)



