import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load model
model = tf.keras.models.load_model("model.h5")

# Title
st.title("Breast Cancer Detection")

# Upload image
file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image")

    # Preprocess
    img = img.resize((160,160))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img)[0][0]

    if pred > 0.5:
        st.error(f"Cancer Detected 😢 ({pred:.2f})")
    else:
        st.success(f"Normal ✅ ({pred:.2f})")
