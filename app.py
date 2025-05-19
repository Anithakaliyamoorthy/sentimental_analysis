import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from langdetect import detect
import speech_recognition as sr
import tempfile
import shap
import torch
import matplotlib.pyplot as plt

# Set page config first
st.set_page_config(page_title="RoBERTa Sentiment Analysis", layout="centered")

st.title("🔍 Sentiment Analyzer with Voice, SHAP & Multilingual Support")
st.markdown("Analyze sentiment from **text** or **audio** with explainability.")

# Load model with caching
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier, tokenizer, model

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

# 1. Text input
st.header("📝 Enter Text")
text_input = st.text_area("Type or paste text below:", placeholder="e.g., I love the product quality!", height=150)

# 2. Audio input
st.header("🎤 Upload Audio File (WAV or MP3)")
audio_file = st.file_uploader("Upload audio file for sentiment analysis", type=["wav", "mp3"])

transcribed_text = ""
if audio_file:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            audio_path = tmp.name

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)

        transcribed_text = recognizer.recognize_google(audio)
        st.success(f"Transcribed Text: {transcribed_text}")
    except Exception as e:
        st.error(f"Audio processing error: {e}")

# Final input to analyze
text_to_analyze = transcribed_text if transcribed_text else text_input

if st.button("🔍 Analyze Sentiment"):
    if text_to_analyze.strip():
        with st.spinner("Analyzing..."):
            # Language detection
            try:
                lang = detect(text_to_analyze)
                st.info(f"🌐 Detected Language: {lang.upper()}")
            except Exception:
                st.info("🌐 Language detection failed, proceeding...")

            # Prediction
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

            # Text plot
            fig_text, ax_text = plt.subplots(figsize=(10, 1))
            shap.plots.text(shap_values[0], display=False)
            st.pyplot(fig_text)
            plt.close(fig_text)

            # Bar plot
            fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
            shap.plots.bar(shap_values[0], max_display=20, show=False)
            st.pyplot(fig_bar)
            plt.close(fig_bar)

            # Waterfall plot for predicted class
            class_idx = int(label_code.split('_')[1])
            single_class_shap_values = shap.Explanation(
                values=shap_values.values[0][:, class_idx],
                base_values=shap_values.base_values[0][class_idx],
                data=shap_values.data[0],
                feature_names=shap_values.feature_names
            )
            fig_wf = plt.figure(figsize=(10, 6))
            shap.waterfall_plot(single_class_shap_values, max_display=20)
            st.pyplot(fig_wf)
            plt.close(fig_wf)

    else:
        st.warning("Please enter text or upload an audio file to analyze.")

st.markdown("<hr><center>Built with 🤖 RoBERTa and Streamlit</center>", unsafe_allow_html=True)

