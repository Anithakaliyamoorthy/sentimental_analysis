import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from langdetect import detect
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import shap
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Set page config early
st.set_page_config(page_title="RoBERTa Sentiment Analysis", layout="centered")

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
st.markdown("This app uses RoBERTa to analyze sentiment from **text** or **audio** with explainability.")

# Text input
st.header("📝 Enter Text")
text_input = st.text_area("Type or paste text below:", placeholder="e.g., I love the product quality!", height=150)

# Audio upload
st.header("🎤 Upload Audio File (WAV or MP3)")
audio_file = st.file_uploader("Upload audio file for sentiment analysis", type=["wav", "mp3"])

transcribed_text = ""
if audio_file:
    try:
        audio_format = audio_file.name.split('.')[-1]
        audio = AudioSegment.from_file(audio_file, format=audio_format)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            audio.export(tmp_wav.name, format="wav")
            audio_path = tmp_wav.name
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            transcribed_text = recognizer.recognize_google(audio_data)
        st.success(f"Transcribed Text: {transcribed_text}")
    except Exception as e:
        st.error("Audio processing error: " + str(e))

text_to_analyze = transcribed_text if transcribed_text else text_input

if st.button("🔍 Analyze Sentiment"):
    if text_to_analyze.strip():
        with st.spinner("Analyzing..."):
            try:
                lang = detect(text_to_analyze)
                st.info(f"🌐 Detected Language: {lang.upper()}")
            except:
                st.warning("Could not detect language.")

            result = classifier(text_to_analyze)[0]
            label_code = result['label']
            label, emoji = label_map[label_code]
            score = result['score'] * 100

            st.success(f"Sentiment: **{label}** {emoji}")
            st.write(f"Confidence Score: **{score:.2f}%**")
            st.info(suggestions[label])

            # SHAP explanation
            st.subheader("📊 SHAP Explanation (Why this prediction?)")

            def f(X):
                inputs = tokenizer(list(X), padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**inputs).logits
                return torch.nn.functional.softmax(logits, dim=1).numpy()

            explainer = shap.Explainer(f, tokenizer)
            shap_values = explainer([text_to_analyze])

          # 1) Text explanation (highlight words)
shap.plots.text(shap_values[0], display=False)
fig1 = plt.gcf()
st.pyplot(fig1)
plt.clf()

# 2) Bar plot of SHAP values per token
st.subheader("📈 SHAP Bar Plot")
tokens = shap_values.data[0]
values = shap_values.values[0, :, int(label_code[-1])]

fig2, ax2 = plt.subplots(figsize=(10, 4))
colors = ['red' if v < 0 else 'green' for v in values]
ax2.bar(range(len(tokens)), values, color=colors)
ax2.set_xticks(range(len(tokens)))
ax2.set_xticklabels(tokens, rotation=45, ha="right")
ax2.set_ylabel("SHAP value")
ax2.set_title("SHAP values per token")
st.pyplot(fig2)
plt.clf()

# 3) Waterfall chart for contribution of tokens to prediction
st.subheader("🌊 SHAP Waterfall Chart")
shap.waterfall_plot(shap_values[0], max_display=20)
fig3 = plt.gcf()
st.pyplot(fig3)
plt.clf()

    else:
        st.warning("Please enter text or upload an audio file to analyze.")

st.markdown("<br><hr style='border:0.5px solid #ccc;'><center>Built with 🤖 RoBERTa and Streamlit</center>", unsafe_allow_html=True)

