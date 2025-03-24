import os
import tensorflow as tf
import numpy as np
import gradio as gr
import requests

# Google Drive File ID
MODEL_FILE_ID = "124ApKoOArzTXfQ8y9Zz0ZKSQT71kbwzt"
MODEL_PATH = "ecg_classification_model.keras"

# Function to download the model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        response = requests.get(url, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Model downloaded successfully!")

# Download model before loading
download_model()

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH, compile=True)

# Class labels
class_labels = [
    "Left Bundle Branch Block",
    "Normal",
    "Premature Atrial Contraction",
    "Premature Ventricular Contractions",
    "Right Bundle Branch Block",
    "Ventricular Fibrillation"
]

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict_ecg(img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f}%)"

# Gradio Interface
iface = gr.Interface(
    fn=predict_ecg,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="ECG Image Classifier",
    description="Upload an ECG image to classify it."
)

# Launch app
iface.launch(share=True)


