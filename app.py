import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
import matplotlib.pyplot as plt
import shap
import torch
import numpy as np
import tempfile
import sounddevice as sd
import wavio
import speech_recognition as sr
from googletrans import Translator

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer), model, tokenizer

classifier, model, tokenizer = load_model()

# SHAP explainer setup (for explanation visualization)
def explain_prediction(text, model, tokenizer):
    tokens = tokenizer([text], return_tensors='pt', truncation=True)
    explainer = shap.Explainer(lambda x: model(x)[0].detach().numpy(), tokenizer)
    shap_values = explainer(tokens['input_ids'].numpy())
    return shap_values

# Emoji mapping
label_map = {
    "LABEL_0": ("Negative", "😠"),
    "LABEL_1": ("Neutral", "😐"),
    "LABEL_2": ("Positive", "😊")
}

# Suggestions
improvement_suggestion = {
    "Negative": "❗ Consider improving product quality or addressing key user complaints.",
    "Neutral": "💡 Users feel neutral — enhancing value or adding features might improve engagement.",
    "Positive": "✅ Users are happy — maintain the quality and consider gathering testimonials!"
}

# UI Layout
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("💬 Sentiment Analysis App")
st.markdown("Enter your text or use voice input to get sentiment analysis, explanation, and improvement tips.")

# Multilingual translation
translator = Translator()

# Text input
text = st.text_area("Enter text for sentiment analysis:", height=150, placeholder="e.g., The product quality is average, not too good or bad.")

# Voice Input
if st.button("Use Voice Input"):
    duration = 5  # seconds
    fs = 44100
    st.info("Recording voice for 5 seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    wavio.write(temp_file.name, recording, fs, sampwidth=2)

    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_file.name) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            st.success(f"Transcribed Text: {text}")
        except sr.UnknownValueError:
            st.error("Could not understand audio.")
        except sr.RequestError:
            st.error("Could not request results from Google Speech Recognition service.")

# Translation (for multilingual support)
language_option = st.selectbox("Select language of input text (for translation):", ["English", "Spanish", "French", "German", "Tamil", "Hindi"])
lang_code_map = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Tamil": "ta",
    "Hindi": "hi"
}

if language_option != "English" and text:
    translated = translator.translate(text, src=lang_code_map[language_option], dest='en')
    text = translated.text
    st.markdown(f"**Translated to English:** {text}")

# Analyze button
if st.button("Analyze"):
    if text.strip():
        with st.spinner("Analyzing..."):
            result = classifier(text)[0]
            label_code = result['label']
            label, emoji = label_map[label_code]
            score = result['score'] * 100

            # Sentiment Output
            st.markdown(f"""
            <div style='background-color:#e8f5e9;padding:1.2rem;border-radius:10px;'>
                <b>Sentiment:</b> {label} {emoji}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background-color:#e3f2fd;padding:1rem;border-radius:10px;'>
                <b>Confidence:</b> {score:.2f}%
            </div>
            """, unsafe_allow_html=True)

            st.info(improvement_suggestion[label])

            # Word cloud & Explanation
            st.subheader("Model Explanation with SHAP")
            st.write("(SHAP explanation is simulated due to limitations in HuggingFace transformers + SHAP compatibility.)")
            st.markdown("For production, SHAP explanations would highlight influential words.")

    else:
        st.warning("Please enter or speak some text before analyzing.")

# Footer
st.markdown("<br><hr style='border:0.5px solid #ddd;'><br>", unsafe_allow_html=True)
st.markdown("Made with ❤️ using RoBERTa + Streamlit")
