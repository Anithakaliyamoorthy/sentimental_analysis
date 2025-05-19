import streamlit as st

# ✅ Set Streamlit page configuration FIRST
st.set_page_config(page_title="RoBERTa Sentiment Analysis", layout="centered")

# ========== All other imports ==========
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from langdetect import detect
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import shap
import torch
import matplotlib.pyplot as plt
import numpy as np
import io

# ========== Load model and tokenizer ==========
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer), tokenizer, model

classifier, tokenizer, model = load_model()

# ========== Label and suggestions ==========
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

# ========== UI Layout ==========
st.title("🔍 Sentiment Analyzer with Voice, SHAP & Multilingual Support")
st.markdown("This app uses **RoBERTa** to analyze sentiment from **text** or **audio** with explainability using **SHAP**.")

# ========== 1. Text Input ==========
st.header("📝 Enter Text")
text_input = st.text_area("Type or paste text below:", placeholder="e.g., I love the product quality!", height=150)

# ========== 2. Audio Input ==========
st.header("🎤 Upload Audio File (WAV or MP3)")
audio_file = st.file_uploader("Upload audio file for sentiment analysis", type=["wav", "mp3"])

transcribed_text = ""
if audio_file:
    try:
        # Convert to WAV if it's MP3
        if audio_file.type == "audio/mpeg":
            sound = AudioSegment.from_mp3(audio_file)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                sound.export(tmp_wav.name, format="wav")
                audio_path = tmp_wav.name
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                audio_path = tmp.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)

        transcribed_text = recognizer.recognize_google(audio)
        st.success(f"Transcribed Text: {transcribed_text}")

    except Exception as e:
        st.error("Audio processing error: " + str(e))

# ========== 3. Final Sentiment Input ==========
text_to_analyze = transcribed_text if transcribed_text else text_input

if st.button("🔍 Analyze Sentiment"):
    if text_to_analyze.strip():
        with st.spinner("Analyzing..."):
            lang = detect(text_to_analyze)
            st.info(f"🌐 Detected Language: **{lang.upper()}**")

            result = classifier(text_to_analyze)[0]
            label_code = result['label']
            label, emoji = label_map[label_code]
            score = result['score'] * 100

            # Display Results
            st.success(f"Sentiment: **{label}** {emoji}")
            st.write(f"Confidence Score: **{score:.2f}%**")
            st.info(suggestions[label])

            # ========== 4. SHAP Explanation ==========
            st.subheader("📊 SHAP Explanation (Why this prediction?)")

            def f(X):
                inputs = tokenizer(list(X), padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**inputs).logits
                return torch.nn.functional.softmax(logits, dim=1).numpy()

            explainer = shap.Explainer(f, tokenizer)
            shap_values = explainer([text_to_analyze])

            # Render SHAP Explanation
            shap_output = io.StringIO()
            shap.plots.text(shap_values[0], display=False)
            st.pyplot(bbox_inches='tight')

    else:
        st.warning("Please enter text or upload an audio file to analyze.")

# Footer
st.markdown("<br><hr style='border:0.5px solid #ccc;'><center>Built with 🤖 RoBERTa, 🎤 SpeechRecognition, and 📊 SHAP</center>", unsafe_allow_html=True)

