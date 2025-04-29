# -*- coding: utf-8 -*-
"""
Created on [Date]

@author: [Your Name]
"""

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

# Load your trained model
model = load_model("pneumonia_model.h5")  # Update path if needed

# Streamlit UI
st.title("Pneumonia Detection from Chest X-ray")
st.markdown("Upload a chest X-ray image to detect if the patient has **Pneumonia** or is **Normal**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Chest X-ray Image", use_column_width=True)

    # Preprocess the image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    result = "🦠 **PNEUMONIA Detected**" if prediction >= 0.5 else "✅ **Normal**"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    # Output
    st.markdown(f"### Prediction: {result}")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
