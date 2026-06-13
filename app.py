import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('model.h5')
IMG_SIZE = 128

def predict(img):
    img = Image.fromarray(img).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        return {"Cat": float(1 - pred), "Dog": float(pred)}
    else:
        return {"Cat": float(1 - pred), "Dog": float(pred)}

gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload a Cat or Dog image"),
    outputs=gr.Label(label="Prediction"),
    title="🐱 Cat vs Dog Classifier 🐶",
    description="CNN model trained with Keras · 89% test accuracy",
).launch()