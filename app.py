import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from langdetect import detect
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import shap
import torch
import matplotlib.pyplot as plt
import os

# Add FFMPEG path if on Windows
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"

st.set_page_config(page_title="RoBERTa Sentiment Analyzer", layout="centered")

@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer), tokenizer, model

classifier, tokenizer, model = load_model()

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

st.title("🔍 Sentiment Analyzer with Voice, SHAP & Multilingual Support")
st.markdown("Analyze sentiment from **text** or **audio** with explainability.")

# Text input
st.header("📝 Enter Text")
text_input = st.text_area("Type or paste text below:", placeholder="e.g., I love the product quality!", height=150)

# Audio input
st.header("🎤 Upload Audio File (WAV or MP3)")
audio_file = st.file_uploader("Upload audio file for sentiment analysis", type=["wav", "mp3"])

transcribed_text = ""
if audio_file:
    try:
        # Convert audio to WAV PCM using pydub
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            audio = AudioSegment.from_file(audio_file)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # PCM 16-bit mono
            audio.export(tmp_wav.name, format="wav")
            audio_path = tmp_wav.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)

        transcribed_text = recognizer.recognize_google(audio_data)
        st.success(f"Transcribed Text: {transcribed_text}")
    except Exception as e:
        st.error(f"Audio processing error: {e}")

# Final text to analyze
text_to_analyze = transcribed_text if transcribed_text else text_input

if st.button("🔍 Analyze Sentiment"):
    if text_to_analyze.strip():
        with st.spinner("Analyzing..."):
            lang = detect(text_to_analyze)
            st.info(f"🌐 Detected Language: {lang.upper()}")

            result = classifier(text_to_analyze)[0]
            label_code = result['label']
            label, emoji = label_map[label_code]
            score = result['score'] * 100

            st.success(f"Sentiment: **{label}** {emoji}")
            st.write(f"Confidence Score: **{score:.2f}%**")
            st.info(suggestions[label])

            # SHAP Explanation
            st.subheader("📊 SHAP Explanation (Why this prediction?)")

            try:
                def f(X):
                    inputs = tokenizer(list(X), padding=True, truncation=True, return_tensors="pt")
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    return torch.nn.functional.softmax(logits, dim=1).numpy()

                explainer = shap.Explainer(f, tokenizer)
                shap_values = explainer([text_to_analyze])

                # Try SHAP text plot
                try:
                    fig_text = plt.figure()
                    shap.plots.text(shap_values[0])
                    st.pyplot(fig_text)
                except Exception:
                    # Fallback to bar plot if text plot fails
                    fig_bar, ax = plt.subplots()
                    shap.plots.bar(shap_values[0], max_display=10, show=False)
                    st.pyplot(fig_bar)

            except Exception as e:
                st.error(f"SHAP visualization error: {e}")
                st.write("Fallback: Showing raw model logits.")
                inputs = tokenizer(text_to_analyze, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    logits = model(**inputs).logits
                st.write(logits.softmax(dim=1))

    else:
        st.warning("Please enter text or upload an audio file to analyze.")

st.markdown("<hr><center>Built with 🤖 RoBERTa and Streamlit</center>", unsafe_allow_html=True)
