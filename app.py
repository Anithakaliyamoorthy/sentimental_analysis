import matplotlib.pyplot as plt
import shap

# ... your existing code above ...

if st.button("🔍 Analyze Sentiment"):
    if text_to_analyze.strip():
        with st.spinner("Analyzing..."):
            lang = detect(text_to_analyze)
            st.info(f"🌐 Detected Language: {lang.upper()}")

            result = classifier(text_to_analyze)[0]
            label_code = result['label']  # e.g., "LABEL_2"
            label, emoji = label_map[label_code]
            score = result['score'] * 100

            # Display Results
            st.success(f"Sentiment: **{label}** {emoji}")
            st.write(f"Confidence Score: **{score:.2f}%**")
            st.info(suggestions[label])

            # ========== SHAP Explanation ==========
            st.subheader("📊 SHAP Explanation (Why this prediction?)")

            # Define prediction function for SHAP (softmax probabilities)
            def f(X):
                inputs = tokenizer(list(X), padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**inputs).logits
                return torch.nn.functional.softmax(logits, dim=1).numpy()

            explainer = shap.Explainer(f, tokenizer)
            shap_values = explainer([text_to_analyze])

            # Text plot
            fig_text = plt.figure(figsize=(10, 1))
            shap.plots.text(shap_values[0], display=False)
            st.pyplot(fig_text)
            plt.clf()

            # Bar plot for all classes (summary)
            fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
            shap.plots.bar(shap_values[0], max_display=20, show=False)
            st.pyplot(fig_bar)
            plt.clf()

            # Waterfall plot for predicted class
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
        st.warning("Please enter text or upload an audio file to analyze.")
