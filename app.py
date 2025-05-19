import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from langdetect import detect
import torch
import shap
import matplotlib.pyplot as plt

# 1. Load model (cache decorator)
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

st.set_page_config(page_title="RoBERTa Sentiment Analysis", layout="centered")
st.title("🔍 Sentiment Analyzer with SHAP Explainability")

text_input = st.text_area("Type or paste text below:")

if st.button("🔍 Analyze Sentiment"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            lang = detect(text_input)
            st.info(f"🌐 Detected Language: {lang.upper()}")

            result = classifier(text_input)[0]
            label_code = result['label']
            label, emoji = label_map[label_code]
            score = result['score'] * 100

            st.success(f"Sentiment: **{label}** {emoji}")
            st.write(f"Confidence Score: **{score:.2f}%**")
            st.info(suggestions[label])

            # SHAP explanation
            def f(X):
                inputs = tokenizer(list(X), padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**inputs).logits
                return torch.nn.functional.softmax(logits, dim=1).numpy()

            explainer = shap.Explainer(f, tokenizer)
            shap_values = explainer([text_input])

            fig_text = plt.figure(figsize=(10, 1))
            shap.plots.text(shap_values[0], display=False)
            st.pyplot(fig_text)
            plt.clf()

            fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
            shap.plots.bar(shap_values[0], max_display=20, show=False)
            st.pyplot(fig_bar)
            plt.clf()

            class_idx = int(label_code.split('_')[1])
            single_class_shap_values = shap.Explanation(
                values=shap_values.values[0][:, class_idx],
                base_values=shap_values.base_values[0][class_idx],
                data=shap_values.data[0],
                feature_names=shap_values.feature_names
            )

            fig_waterfall = plt.figure(figsize=(10, 6))
            shap.waterfall_plot(single_class_shap_values, max_display=20)
            st.pyplot(fig_waterfall)
            plt.clf()

    else:
        st.warning("Please enter some text to analyze.")
