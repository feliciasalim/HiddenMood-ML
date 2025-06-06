# Import library yang diperlukan
from fastapi import FastAPI              # Untuk membuat API
from pydantic import BaseModel           # Untuk validasi input data
import numpy as np                       # Untuk manipulasi array numerik
import pickle                            # Untuk memuat tokenizer yang disimpan
import tensorflow as tf                  # Untuk model deep learning
from tensorflow.keras.preprocessing.sequence import pad_sequences # Untuk padding input teks #type: ignore

# === Konstanta ===
MODEL_PATH = "models/mental_health/LSTM_model.h5"             # Path file model LSTM
TOKENIZER_PATH = "models/mental_health/tokenizer.pkl"        # Path file tokenizer
MAXLEN = 500                             # Panjang maksimum input teks (dalam token)

# === Load model dan tokenizer ===
model = tf.keras.models.load_model(MODEL_PATH)  # Load model LSTM
with open(TOKENIZER_PATH, "rb") as f:           # Load tokenizer
    tokenizer = pickle.load(f)

# === Label mapping untuk hasil prediksi ===
stress_labels = {0: "Low", 1: "Medium", 2: "High"}  # Label stress
emotion_labels = {0: "Anxious", 1: "Lonely", 2: "Depressed", 3: "Overwhelmed", 4: "Panicked"}  # Label emosi

# === Inisialisasi FastAPI ===
app = FastAPI()

# === Schema input untuk endpoint ===
class TextInput(BaseModel):
    text: str  # Input berupa teks

# === Fungsi untuk preprocessing teks ===
def preprocess_text(text, tokenizer, maxlen=MAXLEN):
    sequences = tokenizer.texts_to_sequences([text])  # Ubah teks menjadi urutan token
    return pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')  # Lakukan padding

# === Endpoint untuk prediksi ===
@app.post("/predict")
def predict(data: TextInput):
    text = data.text                                # Ambil teks dari input
    input_seq = preprocess_text(text, tokenizer)    # Preprocessing teks

    prediction = model.predict(input_seq)           # Lakukan prediksi dengan model

    # === Validasi bentuk output model ===
    if isinstance(prediction, np.ndarray):
        prediction = prediction[0]                  # Ambil prediksi pertama
        if prediction.shape[0] != 8:                # Pastikan shape sesuai ekspektasi
            raise ValueError(f"Expected prediction of shape (8,), but got {prediction.shape}")
        
        # Bagi hasil prediksi menjadi 3 untuk stress dan 5 untuk emosi
        stress_logits = prediction[:3]
        emotion_logits = prediction[3:]

    elif isinstance(prediction, list) and len(prediction) == 2:
        # Jika model output berupa dua array terpisah
        stress_logits = prediction[0][0]
        emotion_logits = prediction[1][0]

    else:
        raise ValueError(f"Unexpected prediction type or structure: {type(prediction)}")

    # === Validasi isi logits ===
    if stress_logits.size == 0 or emotion_logits.size == 0:
        raise ValueError("Empty prediction logits received from model.")

    # Ambil indeks label dengan nilai tertinggi (argmax)
    stress_pred = int(np.argmax(stress_logits))
    emotion_pred = int(np.argmax(emotion_logits))

    #stress_confidence = float(np.max(stress_logits))
    #emotion_confidence = float(np.max(emotion_logits))

    # === Output prediksi ke client ===
    return {
        "text": text,
        "predicted_stress": {
            "label": stress_labels.get(stress_pred, "Unknown"),  # Ambil label stress
            #"confidence": stress_confidence
        },
        "predicted_emotion": {
            "label": emotion_labels.get(emotion_pred, "Unknown"),  # Ambil label emosi
            #"confidence": emotion_confidence
        }
    }
