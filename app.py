import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
import io

# Title
st.title("Breast Cancer Detection App")
st.write("Upload an image to get prediction + GradCAM 🔬")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prepare file for API
    files = {"file": uploaded_file.getvalue()}

    # Call API
    response = requests.post(
        "https://breast-cancer-api-1-mx88.onrender.com/predict",
        files=files
    )

    # Debug (optional)
    st.write("API Response:", response.text)

    # Handle response
    if response.status_code == 200:
        result = response.json()

        # Show prediction
        st.subheader("Prediction Result")
        st.write(result["prediction"])

        # 🔥 Decode GradCAM
        gradcam_bytes = bytes.fromhex(result["gradcam"])
        nparr = np.frombuffer(gradcam_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Show GradCAM image
        st.subheader("GradCAM Output")
        st.image(img, use_column_width=True)

    else:
        st.error("API Error ❌")
