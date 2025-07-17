import streamlit as st
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

# Load model and processor with caching
@st.cache_resource
def load_model():
    model_name = "google/vit-base-patch16-224"
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    return model, processor

# Prediction function
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

# Streamlit UI
st.set_page_config(page_title="Image Classifier", page_icon="🧠")
st.title("🧠 Image Classifier")

uploaded_file = st.file_uploader("Upload an image to classify", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model, processor = load_model()
    predictions = classify(image, model, processor)

    st.subheader("Top 5 Predictions:")
    for label, score in predictions.items():
        st.write(f"**{label}**: {score}%")

    # Show predictions as a bar chart
    st.bar_chart(predictions)

    # Image details
    st.caption(f"**Format:** {image.format or 'N/A'} | **Size:** {image.size} | **Mode:** {image.mode}")

    # Download button for results
    result_text = "\n".join(f"{label}: {score}%" for label, score in predictions.items())
    st.download_button("Download Results", result_text, file_name="classification_result.txt")
