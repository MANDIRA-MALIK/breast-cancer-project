import streamlit as st
import requests
from PIL import Image

st.title("Breast Cancer Detection App")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        try:
            response = requests.post(
                "https://breast-cancer-api-1-mx88.onrender.com/predict",
                files={"file": uploaded_file.getvalue()}
            )

            if response.status_code == 200:
                result = response.json()

                if result["prediction"] == "Malignant":
                    st.error("⚠️ Malignant")
                else:
                    st.success("✅ Benign")
            else:
                st.error("API Error")

        except Exception as e:
            st.error(f"Error: {e}")
