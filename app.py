import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
model = load_model('model.h5')

IMG_SIZE = 128

os.makedirs('static/uploads', exist_ok=True)

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    print(f"Raw prediction: {pred:.4f}")

    # pred > 0.5 → Cat, pred < 0.5 → Dog (swapped to match your model)
    if pred > 0.5:
        label = "Cat"
        confidence = pred * 100
    else:
        label = "Dog"
        confidence = (1 - pred) * 100

    return label, round(float(confidence), 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    save_path = os.path.join('static/uploads', file.filename)
    file.save(save_path)
    label, confidence = predict_image(save_path)
    return jsonify({'label': label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)