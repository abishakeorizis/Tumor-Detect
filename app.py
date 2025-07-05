import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import gdown
import os

# ✅ Must be first Streamlit call
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# Download model if not present
MODEL_PATH = "brain_tumor_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=16c4YFgH4HUXRTUm7bk5cRiU01ggr1MlT"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()
IMG_SIZE = model.input_shape[1:3]

st.title("🧠 Brain Tumor Detection Web App")
st.markdown("Upload a brain MRI image to check for tumor presence.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    resized_image = image.resize(IMG_SIZE)
    img_array = img_to_array(resized_image) / 255.0

    # Heuristic 1: color variance
    color_variance = np.std(img_array, axis=(0, 1))

    # Heuristic 2: check if nearly grayscale
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    grayscale_like = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(b - r)) < 0.08

    if np.max(color_variance) > 0.32 or not grayscale_like:
        st.warning("⚠️ Invalid image: Please upload a brain MRI scan.")
    else:
        img = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img)[0][0]
        confidence_tumor = prediction
        confidence_no_tumor = 1 - prediction

        result = "🔴 **Tumor Detected**" if prediction < 0.5 else "🟢 **No Tumor Detected**"

        st.markdown("---")
        st.markdown(f"### Prediction Result: {result}")
        st.markdown(f"**🟢 Confidence (No Tumor):** `{confidence_tumor:.4f}`")
        st.markdown(f"**🔴 Confidence (Tumor):** `{confidence_no_tumor:.4f}`")

        # Pie chart
        st.markdown("### 🔍 Confidence Breakdown")
        labels = ['Tumor', 'No Tumor']
        sizes = [confidence_no_tumor, confidence_tumor]
        colors = ['#ff3333', '#00cc66']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
