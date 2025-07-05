<<<<<<< HEAD
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ✅ Must be first Streamlit call
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# Constants
MODEL_PATH = "brain_tumor_model.h5"

@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# Get expected input size from model
IMG_SIZE = model.input_shape[1:3]

# Streamlit UI
st.title("🧠 Brain Tumor Detection Web App")
st.markdown("Upload a brain MRI image to check for tumor presence.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Ensure image is in RGB format (3 channels)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize and preprocess
    resized_image = image.resize(IMG_SIZE)
    img_array = img_to_array(resized_image) / 255.0

    # Heuristic 1: color variance
    color_variance = np.std(img_array, axis=(0, 1))

    # Heuristic 2: check if nearly grayscale (R≈G≈B)
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    grayscale_like = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(b - r)) < 0.08

    if np.max(color_variance) > 0.32 or not grayscale_like:
        st.warning("⚠️ Invalid image: Please upload a brain MRI scan.")
    else:
        img = np.expand_dims(img_array, axis=0)  # Shape: (1, H, W, 3)

        # Predict
        prediction = model.predict(img)[0][0]
        confidence_tumor = prediction
        confidence_no_tumor = 1 - prediction

        if prediction < 0.5:
            result = "🔴 **Tumor Detected**"
        else:
            result = "🟢 **No Tumor Detected**"

        st.markdown("---")
        st.markdown(f"### Prediction Result: {result}")
        st.markdown(f"**🟢 Confidence (No Tumor):** `{confidence_tumor:.4f}`")
        st.markdown(f"**🔴 Confidence (Tumor):** `{confidence_no_tumor:.4f}`")

        # Show Pie Chart
        st.markdown("### 🔍 Confidence Breakdown")
        labels = ['Tumor', 'No Tumor']
        sizes = [confidence_no_tumor, confidence_tumor]
        colors = ['#ff3333', '#00cc66']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
=======
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ✅ Must be first Streamlit call
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

model_url = "https://drive.google.com/file/d/16c4YFgH4HUXRTUm7bk5cRiU01ggr1MlT/view?usp=drive_link"
gdown.download(model_url, "brain_tumor_model.h5", quiet=False)

@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# Get expected input size from model
IMG_SIZE = model.input_shape[1:3]

# Streamlit UI
st.title("🧠 Brain Tumor Detection Web App")
st.markdown("Upload a brain MRI image to check for tumor presence.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Ensure image is in RGB format (3 channels)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize and preprocess
    resized_image = image.resize(IMG_SIZE)
    img_array = img_to_array(resized_image) / 255.0

    # Heuristic 1: color variance
    color_variance = np.std(img_array, axis=(0, 1))

    # Heuristic 2: check if nearly grayscale (R≈G≈B)
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    grayscale_like = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(b - r)) < 0.08

    if np.max(color_variance) > 0.32 or not grayscale_like:
        st.warning("⚠️ Invalid image: Please upload a brain MRI scan.")
    else:
        img = np.expand_dims(img_array, axis=0)  # Shape: (1, H, W, 3)

        # Predict
        prediction = model.predict(img)[0][0]
        confidence_tumor = prediction
        confidence_no_tumor = 1 - prediction

        if prediction < 0.5:
            result = "🔴 **Tumor Detected**"
        else:
            result = "🟢 **No Tumor Detected**"

        st.markdown("---")
        st.markdown(f"### Prediction Result: {result}")
        st.markdown(f"**🟢 Confidence (No Tumor):** `{confidence_tumor:.4f}`")
        st.markdown(f"**🔴 Confidence (Tumor):** `{confidence_no_tumor:.4f}`")

        # Show Pie Chart
        st.markdown("### 🔍 Confidence Breakdown")
        labels = ['Tumor', 'No Tumor']
        sizes = [confidence_no_tumor, confidence_tumor]
        colors = ['#ff3333', '#00cc66']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
>>>>>>> 94484f80d5fc05badc798c3047e39ab0d591a7fd
