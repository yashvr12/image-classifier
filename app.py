import streamlit as st
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

# App config
st.set_page_config(page_title="Image Classifier", page_icon="🧠", layout="centered")

# Load model
@st.cache_resource
def load_model():
    model_name = "google/vit-base-patch16-224"
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    return model, processor

# Prediction logic
def classify(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top5_probs, top5_indices = torch.topk(probs.squeeze(), k=5)
    return {
        model.config.id2label[idx.item()]: round(prob.item() * 100, 2)
        for idx, prob in zip(top5_indices, top5_probs)
    }

# Sidebar - upload
with st.sidebar:
    st.title("🧠 Image Classifier")
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    st.markdown("---")
    st.info("Model: `google/vit-base-patch16-224`\n\nRecognizes 1000+ object categories.")

# Main area
st.markdown("## 🔎 Classification Results")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        model, processor = load_model()
        predictions = classify(image, model, processor)

    st.subheader("🔝 Top 5 Predictions")
    for label, score in predictions.items():
        st.write(f"- **{label}**: {score}%")

    # Visual bar chart
    st.bar_chart(predictions)

    # File info
    st.caption(f"🖼️ Format: `{image.format or 'N/A'}` | Size: `{image.size}` | Mode: `{image.mode}`")

    # Download result
    result_text = "\n".join(f"{label}: {score}%" for label, score in predictions.items())
    st.download_button("📄 Download Results", result_text, file_name="classification_result.txt")

else:
    st.warning("Please upload an image from the sidebar.")
