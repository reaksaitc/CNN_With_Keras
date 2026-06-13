# 🐱 Cat vs Dog Classifier — CNN with Keras

A Convolutional Neural Network (CNN) image classifier built with TensorFlow/Keras to classify images as **Cat** or **Dog**.

## 🚀 Live Demo

Try the model directly in your browser — no installation required:

👉 **[https://huggingface.co/spaces/reaksaitc/cat-dog-classifier](https://huggingface.co/spaces/reaksaitc/cat-dog-classifier)**

---

## 📊 Model Performance

| Dataset | Accuracy |
|---|---|
| Training | 90.3% |
| Validation | 88.8% |
| Testing | ~89% |

---

## 🏗️ Model Architecture

A Sequential CNN with 3 convolutional blocks:

```
Input (128x128x3)
→ Conv2D + BatchNorm + MaxPooling
→ Conv2D + BatchNorm + MaxPooling
→ Conv2D + BatchNorm + MaxPooling
→ Flatten
→ Dense + Dropout
→ Dense (sigmoid output)
```

---

## 📁 Dataset

- **Source:** Kaggle Dogs vs Cats dataset
- **Total images:** 10,000 (5,000 cats / 5,000 dogs)
- **Split:** 80% train / 10% validation / 10% test
- **Image size:** 128×128 pixels

---

## 🛠️ Training Details

| Parameter | Value |
|---|---|
| Input size | 128×128 |
| Batch size | 32 |
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Callbacks | EarlyStopping, ReduceLROnPlateau |

**Data Augmentation:**
- Rotation, width/height shift
- Shear, zoom, horizontal flip

---

## 💻 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/reaksaitc/CNN_With_Keras.git
cd CNN_With_Keras
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the web app
```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`

---

## 🧰 Tech Stack

- Python 3.12
- TensorFlow / Keras
- NumPy, Pandas
- OpenCV, Pillow
- Scikit-learn
- Flask (local web app)
- Gradio (Hugging Face demo)

---

## 👤 Author

**Reaksai TC**
- GitHub: [@reaksaitc](https://github.com/reaksaitc)
- Hugging Face: [@reaksaitc](https://huggingface.co/reaksaitc)

---

*Institute of Technology of Cambodia — Applied Mathematics and Statistics*
