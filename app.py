import streamlit as st
import requests
import numpy as np
import cv2

st.title("Breast Cancer Detection App")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # API URL (IMPORTANT)
    url = "https://breast-cancer-api-1-mx88.onrender.com/predict"

    # Prepare file
    files = {
        "file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")
    }

    # Send request
    response = requests.post(url, files=files)

    # Debug info (VERY IMPORTANT)
    st.write("Status Code:", response.status_code)
    st.write("Response Text:", response.text)

    # Handle response safely
    if response.status_code == 200:
        try:
            result = response.json()

            st.success("Prediction Done ✅")

            # Prediction
            prediction = result.get("prediction", None)

            if prediction is not None:
                if prediction > 0.5:
                    st.error("⚠️ Malignant (Cancer Detected)")
                else:
                    st.success("✅ Benign (No Cancer)")

            # GradCAM
            if "gradcam" in result:
                gradcam_bytes = bytes.fromhex(result["gradcam"])
                nparr = np.frombuffer(gradcam_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                st.image(img, caption="GradCAM Heatmap 🔥")

        except Exception as e:
            st.error(f"JSON Error: {e}")

    else:
        st.error("API Failed ❌")
