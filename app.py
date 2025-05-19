import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
import whisper
import tempfile
import shap
import torch
import matplotlib.pyplot as plt
import pandas as pd
from langdetect import detect
from reportlab.pdfgen import canvas
import base64
from io import BytesIO

# Set page config first
st.set_page_config(page_title="RoBERTa Sentiment Analyzer", layout="centered")

# Load model
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
    "Negative": "❗ Improve service or address negative feedback.",
    "Neutral": "💡 Increase engagement or content value.",
    "Positive": "✅ Maintain quality and encourage feedback!"
}

st.title("🔍 Sentiment Analyzer (Text + Voice) with SHAP, Export & More")
st.markdown("Analyze text/audio sentiment using RoBERTa with explainability and export options.")

# Text Input
st.header("📝 Enter Text")
text_input = st.text_area("Type or paste text below:", placeholder="e.g., The service was amazing!", height=150)

# Audio Upload
st.header("🎤 Upload Audio File (WAV, MP3, M4A)")
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
transcribed_text = ""

if audio_file:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        model_whisper = whisper.load_model("base")
        result = model_whisper.transcribe(tmp_path)
        transcribed_text = result["text"]
        st.success(f"Transcribed: {transcribed_text}")
    except Exception as e:
        st.error(f"Audio processing error: {e}")

text_to_analyze = transcribed_text if transcribed_text else text_input

if st.button("🔍 Analyze Sentiment"):
    if text_to_analyze.strip():
        with st.spinner("Analyzing..."):
            lang = detect(text_to_analyze)
            st.info(f"🌍 Language detected: {lang.upper()}")

            result = classifier(text_to_analyze)[0]
            label_code = result['label']
            label, emoji = label_map[label_code]
            score = result['score'] * 100

            st.success(f"Sentiment: **{label}** {emoji}")
            st.write(f"Confidence: **{score:.2f}%**")
            st.info(suggestions[label])

            # SHAP Explainability
            st.subheader("📊 SHAP Explanation (Bar Plot)")

            def predict_fn(X):
                inputs = tokenizer(list(X), padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                return torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()

            explainer = shap.Explainer(predict_fn, tokenizer)
            shap_values = explainer([text_to_analyze])

            fig, ax = plt.subplots(figsize=(10, 4))
            shap.plots.bar(shap_values[0], show=False)
            st.pyplot(fig)

            # Download CSV
            st.subheader("📥 Download Result as CSV")
            df = pd.DataFrame([{
                "Text": text_to_analyze,
                "Sentiment": label,
                "Confidence": f"{score:.2f}%",
                "Suggestion": suggestions[label]
            }])
            csv = df.to_csv(index=False).encode()
            st.download_button("Download CSV", csv, "sentiment_result.csv", "text/csv")

            # Download PDF
            st.subheader("📄 Download Result as PDF")
            buffer = BytesIO()
            c = canvas.Canvas(buffer)
            c.drawString(100, 800, f"Sentiment Analysis Report")
            c.drawString(100, 780, f"Text: {text_to_analyze[:80]}...")
            c.drawString(100, 760, f"Sentiment: {label} ({emoji})")
            c.drawString(100, 740, f"Confidence: {score:.2f}%")
            c.drawString(100, 720, f"Suggestion: {suggestions[label]}")
            c.save()
            pdf = buffer.getvalue()
            b64_pdf = base64.b64encode(pdf).decode()
            st.markdown(f'<a href="data:application/pdf;base64,{b64_pdf}" download="sentiment_report.pdf">📥 Download PDF Report</a>', unsafe_allow_html=True)
    else:
        st.warning("Please provide text or upload audio first.")

st.markdown("<hr><center>Built with 🤗 Transformers, Whisper & Streamlit</center>", unsafe_allow_html=True)

