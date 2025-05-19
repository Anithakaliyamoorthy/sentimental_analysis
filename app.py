import streamlit as st
from transformers import pipeline
from langdetect import detect
import speech_recognition as sr
import tempfile
import matplotlib.pyplot as plt

# Set page config (must be first Streamlit command)
st.set_page_config(page_title="🎙️ Sentiment Analyzer", layout="centered")

# Load model once and cache
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

classifier = load_model()

label_map = {
    "LABEL_0": ("Negative", "😠"),
    "LABEL_1": ("Neutral", "😐"),
    "LABEL_2": ("Positive", "😊")
}

suggestions = {
    "Negative": "❗ Consider addressing the issues raised by users.",
    "Neutral": "💡 Try to make content more engaging.",
    "Positive": "✅ Keep up the great work and engage your happy users!"
}

st.title("🔍 Sentiment Analyzer (Text + Voice Input)")
st.markdown("Analyze sentiment from **text** or **voice** using RoBERTa and Google Speech Recognition.")

# Text input
st.header("📝 Enter Text")
text_input = st.text_area("Type your input:", placeholder="e.g., I love this product!", height=150)

# Audio input
st.header("🎤 Upload Audio File (WAV only)")
audio_file = st.file_uploader("Choose a WAV file", type=["wav"])

transcribed_text = ""
if audio_file:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)
        transcribed_text = recognizer.recognize_google(audio)
        st.success(f"Transcribed Audio: `{transcribed_text}`")
    except Exception as e:
        st.error(f"Audio processing error: {e}")

# Choose text to analyze
text_to_analyze = transcribed_text or text_input

if st.button("🔍 Analyze Sentiment"):
    if not text_to_analyze.strip():
        st.warning("Please provide text or upload audio.")
    else:
        with st.spinner("Analyzing..."):
            try:
                lang = detect(text_to_analyze)
                st.info(f"🌐 Detected Language: `{lang.upper()}`")

                result = classifier(text_to_analyze)[0]
                label_code = result["label"]
                label, emoji = label_map.get(label_code, ("Unknown", "❓"))
                confidence = result["score"] * 100

                st.success(f"Sentiment: **{label}** {emoji}")
                st.write(f"Confidence: **{confidence:.2f}%**")
                st.info(suggestions.get(label, "No suggestion."))

                # Visualize with bar chart
                st.subheader("📊 Sentiment Confidence")
                fig, ax = plt.subplots()
                ax.bar([label], [confidence], color="green" if label == "Positive" else "orange" if label == "Neutral" else "red")
                ax.set_ylim(0, 100)
                ax.set_ylabel("Confidence (%)")
                ax.set_title("Sentiment Confidence Score")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error during analysis: {e}")

st.markdown("<hr><center>Built with 🤖 RoBERTa + 🎤 Google Speech + 📊 Matplotlib</center>", unsafe_allow_html=True)
