import streamlit as st
import requests
import tempfile
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from langdetect import detect
import plotly.graph_objects as go

# AssemblyAI API Key
ASSEMBLYAI_API_KEY = "38c7d7ea134e44c9b073c0454978b3ee"

# Load RoBERTa model
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer), tokenizer, model

classifier, tokenizer, model = load_model()

# Label map for results
label_map = {
    "LABEL_0": ("Negative", "😠"),
    "LABEL_1": ("Neutral", "😐"),
    "LABEL_2": ("Positive", "😊")
}

suggestions = {
    "Negative": "❗ Improve service or content based on negative feedback.",
    "Neutral": "💡 Try to make content more engaging.",
    "Positive": "✅ Keep up the good work and ask users for more feedback!"
}

st.set_page_config(page_title="🎙️ Sentiment Analyzer", layout="centered")
st.title("🔍 RoBERTa Sentiment Analyzer with Voice & Visualization")
st.markdown("Analyze sentiment from **text** or **voice** and visualize the result interactively.")

# Text Input
st.header("📝 Enter Text")
text_input = st.text_area("Type or paste your text below:", height=150)

# Audio Input
st.header("🎤 Upload Audio File")
audio_file = st.file_uploader("Upload a WAV or MP3 audio file", type=["wav", "mp3"])

transcribed_text = ""
if audio_file:
    st.info("Transcribing audio using AssemblyAI...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    # Upload to AssemblyAI
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    upload_res = requests.post("https://api.assemblyai.com/v2/upload",
                               headers=headers, data=open(tmp_path, "rb"))

    if upload_res.status_code == 200:
        audio_url = upload_res.json()["upload_url"]
        transcript_res = requests.post("https://api.assemblyai.com/v2/transcript",
                                       headers=headers,
                                       json={"audio_url": audio_url})
        transcript_id = transcript_res.json()["id"]

        # Poll status
        status = "processing"
        with st.spinner("Processing transcription..."):
            while status != "completed":
                poll_res = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                                        headers=headers)
                status = poll_res.json()["status"]
                if status == "completed":
                    transcribed_text = poll_res.json()["text"]
                elif status == "error":
                    st.error("❌ Transcription failed.")
                    break
    else:
        st.error("❌ Audio upload failed.")

if transcribed_text:
    st.success(f"🗣️ Transcribed Text: {transcribed_text}")

# Decide final text
final_text = transcribed_text if transcribed_text else text_input

# Analyze Sentiment
if st.button("🔍 Analyze Sentiment"):
    if final_text.strip():
        with st.spinner("Analyzing sentiment..."):
            try:
                lang = detect(final_text)
                st.info(f"🌐 Detected Language: {lang.upper()}")
            except:
                st.warning("⚠️ Could not detect language.")

            result = classifier(final_text)[0]
            label_code = result['label']
            label, emoji = label_map[label_code]
            score = result['score'] * 100

            st.success(f"Sentiment: **{label}** {emoji}")
            st.write(f"Confidence Score: **{score:.2f}%**")
            st.info(suggestions[label])

            # 🌟 Plotly Visualization
            st.subheader("📊 Visualize Sentiment Scores")
            inputs = tokenizer(final_text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

            labels = ["Negative", "Neutral", "Positive"]
            fig = go.Figure(go.Bar(
                x=labels,
                y=probs,
                marker_color=["crimson", "gray", "limegreen"]
            ))
            fig.update_layout(title="Sentiment Score Distribution", yaxis_title="Probability", xaxis_title="Sentiment")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please enter text or upload an audio file first.")

st.markdown("---")
st.caption("Built with 🤖 RoBERTa, 🧠 AssemblyAI, 📊 Plotly & Streamlit")
