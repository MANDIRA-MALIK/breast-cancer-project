import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("model.h5")

# 🔥 CHANGE THIS (last conv layer name)
last_conv_layer_name = "Conv_1"   # MobileNetV2 ke liye

# GradCAM function
def get_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# Title
st.title("Breast Cancer Detection + GradCAM")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image")

    # Preprocess
    img_resized = img.resize((160,160))
    img_array = np.array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        st.error(f"Cancer Detected 😢 ({pred:.2f})")
    else:
        st.success(f"Normal ✅ ({pred:.2f})")

    # 🔥 GradCAM
    heatmap = get_gradcam_heatmap(img_array, model, last_conv_layer_name)

    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = heatmap * 0.4 + np.array(img)

    st.image(superimposed.astype("uint8"), caption="GradCAM Output")
