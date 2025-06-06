import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences  
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mental_health", "LSTM_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "mental_health", "tokenizer.pkl")
MAXLEN = 500

model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

stress_labels = {0: "Low", 1: "Medium", 2: "High"}
emotion_labels = {0: "Anxious", 1: "Lonely", 2: "Depressed", 3: "Overwhelmed", 4: "Panicked"}

def preprocess_text(text, tokenizer=tokenizer, maxlen=MAXLEN):
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

def predict_stress_emotion(text):
    input_seq = preprocess_text(text)

    prediction = model.predict(input_seq, verbose=0)

    if isinstance(prediction, np.ndarray):
        prediction = prediction[0]
        if prediction.shape[0] != 8:
            raise ValueError(f"Expected prediction of shape (8,), but got {prediction.shape}")
        stress_logits = prediction[:3]
        emotion_logits = prediction[3:]
    elif isinstance(prediction, list) and len(prediction) == 2:
        stress_logits = prediction[0][0]
        emotion_logits = prediction[1][0]
    else:
        raise ValueError(f"Unexpected prediction type or structure: {type(prediction)}")

    if stress_logits.size == 0 or emotion_logits.size == 0:
        raise ValueError("Empty prediction logits received from model.")

    stress_pred = int(np.argmax(stress_logits))
    emotion_pred = int(np.argmax(emotion_logits))

    return {
        "predicted_stress": {"label": stress_labels.get(stress_pred, "Unknown")},
        "predicted_emotion": {"label": emotion_labels.get(emotion_pred, "Unknown")},
        "stress_logits": stress_logits,
    }

def calculate_stress_level(text_input, vectorized_input, stress_logits):
    stress_probs = stress_logits

    stress_score = stress_probs[0] * 0 + stress_probs[1] * 50 + stress_probs[2] * 100

    low_words = {
        "calm", "okay", "fine", "tire", "bore", "good", "sleepy", "irritate",
        "down", "unmotivate", "lazy", "dull", "frustrate", "annoy", "slightly", "upset",
        "restless", "uneasy", "discontent", "displease"
    }
    med_words = {
        "worry", "anxious", "exhaust", "fatigue", "sadness", "disgust", "disappoint",
        "miserable", "numb", "scare", "terrify", "stress", "anxiety", "cry", "helpless",
        "lose", "motivation", "sleep", "overstress", "pressure", "trigger", "overwhelm",
        "tense", "fearful", "panic", "unsettle", "concern", "distress", "worried", "cant breathe"
    }
    high_words = {
        "worthless", "suicide", "die", "depress", "depression", "isolate", "panic",
        "breakdown", "suffer", "despair", "hopeless", "gaslight", "abuse", "self",
        "harm", "kill", "sick", "ugly", "insecure", "insecurity", "grief", "disorder",
        "assault", "guilt", "paranoia", "nightmare", "reject", "miserable",
        "traumatize", "ptsd", "psychotic", "homicidal", "suicidal", "delusional",
        "cripple", "break", "victimize", "devastate", "abandon"
    }

    def word_based_score(text):
        words = set(text.lower().split())
        low_count = len(words & low_words)
        med_count = len(words & med_words)
        high_count = len(words & high_words)
        total = low_count + med_count + high_count
        if total == 0:
            return 0
        return (high_count * 100 + med_count * 50 + low_count * 0) / total

    combined_score = 0.8 * stress_score + 0.2 * word_based_score(text_input)
    return round(combined_score, 2)
