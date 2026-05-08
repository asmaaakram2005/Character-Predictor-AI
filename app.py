from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import tensorflow as tf
import numpy as np
import joblib
import io
import base64

from PIL import Image, ImageOps

# ==========================================
# FastAPI
# ==========================================
app = FastAPI()

# ==========================================
# Static Files
# ==========================================
app.mount("/minst_model/css", StaticFiles(directory="css"), name="css")
app.mount("/minst_model/js", StaticFiles(directory="js"), name="js")

# ==========================================
# CORS
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Load Models
# ==========================================
print("Loading models...")

ann_model = tf.keras.models.load_model("typography_model.keras")

cnn_model = tf.keras.models.load_model("typography_model_cnn.keras")

label_encoder = joblib.load("label_encoder.joblib")

print("Models loaded successfully ✅")

# ==========================================
# Home
# ==========================================
@app.get("/")
def home():
    return FileResponse("html/Ai_Character_Predictor_Frontend.html")

# ==========================================
# Image Preprocessing
# ==========================================
def preprocess_image(image_bytes):

    # Open image
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Convert to array
    image_array = np.array(image)

    # ==========================================
    # Auto Invert Detection
    # ==========================================
    white_pixels = np.sum(image_array > 127)

    black_pixels = np.sum(image_array <= 127)

    # If background is white -> invert
    if white_pixels > black_pixels:

        image_array = 255 - image_array

    # ==========================================
    # Remove noise
    # ==========================================
    image_array = np.where(image_array > 30, 255, 0).astype("uint8")

    # ==========================================
    # Crop character only
    # ==========================================
    coords = np.argwhere(image_array > 30)

    if coords.size > 0:

        y0, x0 = coords.min(axis=0)

        y1, x1 = coords.max(axis=0) + 1

        image_array = image_array[y0:y1, x0:x1]

    # Back to image
    image = Image.fromarray(image_array)

    # ==========================================
    # Add padding
    # ==========================================
    image = ImageOps.expand(image, border=35, fill=0)

    # ==========================================
    # Resize while keeping aspect ratio
    # ==========================================
    image = ImageOps.pad(
        image,
        (28, 28),
        color=0
    )

    # ==========================================
    # Final array
    # ==========================================
    image_array = np.array(image).astype("float32") / 255.0

    # ==========================================
    # Create Preview
    # ==========================================
    preview_img = Image.fromarray(
        (image_array * 255).astype("uint8")
    )

    buffered = io.BytesIO()

    preview_img.save(buffered, format="PNG")

    img_str = base64.b64encode(
        buffered.getvalue()
    ).decode()

    img_preview = f"data:image/png;base64,{img_str}"

    # ==========================================
    # ANN Input
    # ==========================================
    ann_input = image_array.reshape(1, 784)

    # ==========================================
    # CNN Input
    # ==========================================
    cnn_input = image_array.reshape(1, 28, 28, 1)

    return ann_input, cnn_input, img_preview

# ==========================================
# Prediction Helper
# ==========================================
def predict_model(model, image, model_type):

    predictions = model.predict(image)[0]

    predictions = predictions / np.sum(predictions)

    predicted_class = np.argmax(predictions)

    confidence = float(
        predictions[predicted_class] * 100
    )

    character_label = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    # ==========================================
    # Top 5 Predictions
    # ==========================================
    top_5_indices = np.argsort(predictions)[-5:][::-1]

    top_5_list = []

    for i in top_5_indices:

        top_5_list.append({
            "label": str(
                label_encoder.inverse_transform([i])[0]
            ),
            "confidence": round(
                float(predictions[i] * 100),
                1
            )
        })

    return {
        "prediction": str(character_label),
        "confidence": round(confidence, 2),
        "model": model_type,
        "top_5": top_5_list
    }

# ==========================================
# Predict Endpoint
# ==========================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form(...)
):

    image_bytes = await file.read()

    ann_input, cnn_input, img_preview = preprocess_image(
        image_bytes
    )

    result = {}

    # ==========================================
    # ANN
    # ==========================================
    if model_type == "ann":

        result = predict_model(
            ann_model,
            ann_input,
            "ANN"
        )

    # ==========================================
    # CNN
    # ==========================================
    elif model_type == "cnn":

        result = predict_model(
            cnn_model,
            cnn_input,
            "CNN"
        )

    # ==========================================
    # Compare
    # ==========================================
    elif model_type == "compare":

        ann_result = predict_model(
            ann_model,
            ann_input,
            "ANN"
        )

        cnn_result = predict_model(
            cnn_model,
            cnn_input,
            "CNN"
        )

        result = {
            "ann": ann_result,
            "cnn": cnn_result
        }

    else:

        return {
            "error": "Invalid model type"
        }

    # ==========================================
    # Add processed preview
    # ==========================================
    result["processed_image"] = img_preview

    return result