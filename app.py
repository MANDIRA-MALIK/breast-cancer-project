import streamlit as st
import requests
from PIL import Image
import io

st.title("Breast Cancer Detection + GradCAM")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    st.image(file)

    # API URL (yaha apna Render API link daalo)
    url = "https://your-api-url.onrender.com/predict"

    files = {"file": file.getvalue()}

    res = requests.post(url, files=files)

    result = res.json()

    st.write("Prediction:", result["prediction"])

    if "gradcam" in result:
        image_bytes = bytes(result["gradcam"])
        img = Image.open(io.BytesIO(image_bytes))
        st.image(img, caption="GradCAM")
