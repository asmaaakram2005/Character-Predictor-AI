import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("typography_model.keras")

st.title("Character Predictor AI")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))
    arr = np.array(image) / 255.0
    arr = arr.reshape(1, 28, 28, 1)

    prediction = model.predict(arr)

    st.image(image)
    st.write("Prediction:", np.argmax(prediction))
