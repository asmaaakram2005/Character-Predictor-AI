from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import base64
import tensorflow as tf
import joblib 
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/minst_model/css", StaticFiles(directory="css"), name="css")
app.mount("/minst_model/js", StaticFiles(directory="js"), name="js")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ann_model = tf.keras.models.load_model("typography_model.keras")
cnn_model = tf.keras.models.load_model("typography_model_cnn.keras")

label_encoder = joblib.load("label_encoder.joblib")

@app.get("/")
def home():
    return FileResponse("html/Ai_Character_Predictor_Frontend.html")

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image_array = np.array(image)
    
    if np.mean(image_array) > 127:
        image_array = 255 - image_array
    image_array = np.where(image_array > 20, 255, 0).astype('uint8')
    
    # Save a preview to send back to frontend
    preview_img = Image.fromarray(image_array)
    buffered = io.BytesIO()
    preview_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    img_preview = f"data:image/png;base64,{img_str}"
    
    image_array = image_array.astype('float32') / 255.0

    ann_input = image_array.reshape(1, 784)
    cnn_input = image_array.reshape(1, 28, 28, 1)

    return ann_input, cnn_input, img_preview

def predict_model(model, image, model_type):
  
    
    predictions = model.predict(image)[0]
    
  
    predicted_class = np.argmax(predictions)
    confidence = float(predictions[predicted_class] * 100)
    character_label = label_encoder.inverse_transform([predicted_class])[0]


    top_5_indices = np.argsort(predictions)[-5:][::-1]
    
    top_5_list = []
    for i in top_5_indices:
        top_5_list.append({
            "label": str(label_encoder.inverse_transform([i])[0]),
            "confidence": round(float(predictions[i] * 100), 1)
        })

   
    return {
        "prediction": str(character_label),
        "confidence": round(confidence, 2),
        "model": model_type,
        "top_5": top_5_list  
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Form(...)
):
    image_bytes = await file.read()
    ann_input, cnn_input, img_preview = preprocess_image(image_bytes)

    result = {}
    if model_type == "ann":
        result = predict_model(ann_model, ann_input, "ANN")
    elif model_type == "cnn":
        result = predict_model(cnn_model, cnn_input, "CNN")
    elif model_type == "compare":
        ann_result = predict_model(ann_model, ann_input, "ANN")
        cnn_result = predict_model(cnn_model, cnn_input, "CNN")
        result = {"ann": ann_result, "cnn": cnn_result}
    else:
        return {"error": "Invalid model type"}

    # Add the processed image to the response so the user can see it
    if isinstance(result, dict):
        result["processed_image"] = img_preview
    
    return result