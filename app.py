"""
app.py
------
Streamlit GUI for the Handwritten Character Recognition project.

Users can:
  - Upload an image of a handwritten digit, OR
  - Draw a digit directly on an interactive canvas

The app then predicts the digit, shows a confidence score, and a
probability bar chart across all 10 classes.

Run with:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from predict import load_model, predict_digit

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✏️",
    layout="centered",
)

st.title("✏️ Handwritten Character Recognition")
st.write(
    "A Convolutional Neural Network (CNN) built with PyTorch and "
    "trained on the MNIST dataset predicts handwritten digits (0-9). "
    "Upload an image or draw a digit below to see it in action."
)


@st.cache_resource
def get_model():
    return load_model()


try:
    get_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error(
        "Trained model not found. Please run `python train.py` first to "
        "generate `models/mnist_cnn.pt`."
    )

tab_upload, tab_draw = st.tabs(["📁 Upload Image", "🖌️ Draw Digit"])

image_to_predict = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a handwritten digit image", type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)
        st.image(image_to_predict, caption="Uploaded Image", width=200)

with tab_draw:
    st.write("Draw a single digit (0-9) below:")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None and canvas_result.image_data.any():
        drawn_array = canvas_result.image_data.astype("uint8")
        image_to_predict = Image.fromarray(drawn_array).convert("RGB")

st.divider()

if st.button("🔮 Predict", type="primary", disabled=not model_loaded):
    if image_to_predict is None:
        st.warning("Please upload an image or draw a digit first.")
    else:
        label, confidence, probs = predict_digit(image_to_predict)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Number", label)
        with col2:
            st.metric("Confidence Score", f"{confidence * 100:.2f}%")

        st.subheader("Probability Chart")
        prob_df = pd.DataFrame({
            "Digit": [str(i) for i in range(10)],
            "Probability": probs,
        }).set_index("Digit")
        st.bar_chart(prob_df)

st.divider()
st.caption("CodeAlpha Machine Learning Internship — Handwritten Character Recognition")
