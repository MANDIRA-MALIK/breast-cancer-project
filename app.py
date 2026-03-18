import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
import time

st.title("Breast Cancer Detection App")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    files = {"file": uploaded_file.getvalue()}

    # small delay (Render wakeup)
    time.sleep(2)

    response = requests.post(
        "https://breast-cancer-api-1-mx88.onrender.com/predict",
        files=files
    )

    # DEBUG
    st.write("Status Code:", response.status_code)
    st.write("Raw Response:", response.text)

    if response.status_code == 200:
        try:
            result = response.json()

            # Prediction
            st.subheader("Prediction")
            st.write(result["prediction"])

            # GradCAM
            gradcam_bytes = bytes.fromhex(result["gradcam"])
            nparr = np.frombuffer(gradcam_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            st.subheader("GradCAM Output")
            st.image(img)

        except Exception as e:
            st.error(f"JSON Error: {e}")

    else:
        st.error("API failed ❌")
