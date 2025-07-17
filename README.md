# 🧠 Image Classifier

A simple web-based image classification app built using Streamlit and Hugging Face's Vision Transformer (ViT). Upload any image, and the model will predict the top 5 objects it sees — along with their confidence scores.

---

## 🚀 Features

- 🔍 Classifies uploaded images using ViT (Vision Transformer)
- 📊 Shows top 5 predictions with confidence percentages
- 📈 Visualizes results using a bar chart
- 📥 Allows result download as `.txt` file
- ⚡ Fast and responsive web interface powered by Streamlit

---

## 🛠 Tech Stack

- [Streamlit](https://streamlit.io/) – Frontend UI
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) – Vision Transformer Model
- [Torch (PyTorch)](https://pytorch.org/) – Inference
- [Pillow (PIL)](https://python-pillow.org/) – Image processing

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/image-classifier.git
cd image-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
