import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences  
import os
import logging
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mental_health", "LSTM_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "mental_health", "tokenizer.pkl")
MAXLEN = 500

model = None
tokenizer = None
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def load_model():
    global model
    if model is None:
        try:
            logger.info("Loading TensorFlow model from %s", MODEL_PATH)
            if not os.path.exists(MODEL_PATH):
                logger.error("Model file not found: %s", MODEL_PATH)
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("TensorFlow model loaded successfully")
        except Exception as e:
            logger.error("Failed to load model: %s", str(e), exc_info=True)
            raise
    return model

def load_tokenizer():
    global tokenizer
    if tokenizer is None:
        try:
            logger.info("Loading tokenizer from %s", TOKENIZER_PATH)
            if not os.path.exists(TOKENIZER_PATH):
                logger.error("Tokenizer file not found: %s", TOKENIZER_PATH)
                raise FileNotFoundError(f"Tokenizer file not found: {TOKENIZER_PATH}")
            with open(TOKENIZER_PATH, "rb") as f:
                tokenizer = pickle.load(f)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error("Failed to load tokenizer: %s", str(e), exc_info=True)
            raise
    return tokenizer

def clean_text(text):
    try:
        text = re.sub(r'@[A-Za-z0-9]+', '', text)
        text = re.sub(r'#[A-Za-z0-9]+', '', text) 
        text = re.sub(r'RT[\s]', '', text)  
        text = re.sub(r"http\S+", '', text) 
        text = re.sub(r'[0-9]+', '', text) 
        text = re.sub(r'[^\w\s]', '', text)  
        text = text.replace('\n', ' ')
        text = text.translate(str.maketrans('', '', string.punctuation))  
        text = text.strip() 
        text = re.sub(r'\s+', ' ', text)  
        logger.info("Text cleaned successfully")
        return text
    except Exception as e:
        logger.error("Text cleaning failed: %s", str(e), exc_info=True)
        raise

def lowercase_text(text):
    try:
        text = text.lower()
        logger.info("Text converted to lowercase")
        return text
    except Exception as e:
        logger.error("Lowercase conversion failed: %s", str(e), exc_info=True)
        raise

def lemmatize_text(text):
    try:
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        lemmatized_text = ' '.join(words)
        logger.info("Text lemmatized: %s", lemmatized_text)
        return lemmatized_text
    except Exception as e:
        logger.error("Lemmatization failed: %s", str(e), exc_info=True)
        raise

def tokenize_text(text):
    try:
        tokens = word_tokenize(text)
        logger.info("Text tokenized: %s", tokens)
        return tokens
    except Exception as e:
        logger.error("Tokenization failed: %s", str(e), exc_info=True)
        raise

def detokenize_text(tokens):
    try:
        text = ' '.join(tokens)
        logger.info("Tokens detokenized: %s", text)
        return text
    except Exception as e:
        logger.error("Detokenization failed: %s", str(e), exc_info=True)
        raise

def preprocess_text(text, tokenizer=None, maxlen=MAXLEN):
    try:
        logger.info("Preprocessing text: %s", text)
        text = clean_text(text)
        text = lowercase_text(text)
        text = lemmatize_text(text)
        tokens = tokenize_text(text)
        text = detokenize_text(tokens)
        
        if tokenizer is None:
            tokenizer = load_tokenizer()
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
        logger.info("Text preprocessing successful")
        return padded
    except Exception as e:
        logger.error("Text preprocessing failed: %s", str(e), exc_info=True)
        raise

stress_labels = {0: "Low", 1: "Medium", 2: "High"}
emotion_labels = {0: "Anxious", 1: "Lonely", 2: "Depressed", 3: "Overwhelmed", 4: "Panicked"}

def predict_stress_emotion(text):
    try:
        input_seq = preprocess_text(text)
        
        logger.info("Predicting stress and emotion")
        model = load_model()
        prediction = model.predict(input_seq, verbose=0)

        if isinstance(prediction, np.ndarray):
            prediction = prediction[0]
            if prediction.shape[0] != 8:
                logger.error("Expected prediction of shape (8,), but got %s", prediction.shape)
                raise ValueError(f"Expected prediction of shape (8,), but got {prediction.shape}")
            stress_logits = prediction[:3]
            emotion_logits = prediction[3:]
        elif isinstance(prediction, list) and len(prediction) == 2:
            stress_logits = prediction[0][0]
            emotion_logits = prediction[1][0]
        else:
            logger.error("Unexpected prediction type or structure: %s", type(prediction))
            raise ValueError(f"Unexpected prediction type or structure: {type(prediction)}")

        if stress_logits.size == 0 or emotion_logits.size == 0:
            logger.error("Empty prediction logits received from model")
            raise ValueError("Empty prediction logits received from model.")

        stress_pred = int(np.argmax(stress_logits))
        emotion_pred = int(np.argmax(emotion_logits))

        logger.info("Prediction completed: stress=%s, emotion=%s", stress_labels.get(stress_pred, "Unknown"), emotion_labels.get(emotion_pred, "Unknown"))
        return {
            "predicted_stress": {"label": stress_labels.get(stress_pred, "Unknown")},
            "predicted_emotion": {"label": emotion_labels.get(emotion_pred, "Unknown")},
            "stress_logits": stress_logits.tolist(),
        }
    except Exception as e:
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        raise

def calculate_stress_level(text_input, vectorized_input, stress_logits):
    try:
        stress_probs = stress_logits
        stress_score = stress_probs[0] * 0 + stress_probs[1] * 50 + stress_probs[2] * 100

        low_words = {lemmatizer.lemmatize(w) for w in [
            "calm", "okay", "fine", "tire", "bore", "good", "sleepy", "irritate",
            "down", "unmotivate", "lazy", "dull", "frustrate", "annoy", "slightly", "upset",
            "restless", "uneasy", "discontent", "displease"
        ]}
        med_words = {lemmatizer.lemmatize(w) for w in [
            "worry", "anxious", "exhaust", "fatigue", "sadness", "disgust", "disappoint",
            "miserable", "numb", "scare", "terrify", "stress", "anxiety", "cry", "helpless",
            "lose", "motivation", "sleep", "overstress", "pressure", "trigger", "overwhelm",
            "tense", "fearful", "panic", "unsettle", "concern", "distress", "worried", "cant breathe"
        ]}
        high_words = {lemmatizer.lemmatize(w) for w in [
            "worthless", "suicide", "die", "depress", "depression", "isolate", "panic",
            "breakdown", "suffer", "despair", "hopeless", "gaslight", "abuse", "self",
            "harm", "kill", "sick", "ugly", "insecure", "insecurity", "grief", "disorder",
            "assault", "guilt", "paranoia", "nightmare", "reject", "miserable",
            "traumatize", "ptsd", "psychotic", "homicidal", "suicidal", "delusional",
            "cripple", "break", "victimize", "devastate", "abandon"
        ]}

        def word_based_score(text):
            text = clean_text(text)
            text = lowercase_text(text)
            text = lemmatize_text(text)
            words = set(text.lower().split())
            low_count = len(words & low_words)
            med_count = len(words & med_words)
            high_count = len(words & high_words)
            total = low_count + med_count + high_count
            if total == 0:
                return 0
            return (high_count * 100 + med_count * 50 + low_count * 0) / total

        combined_score = 0.8 * stress_score + 0.2 * word_based_score(text_input)
        logger.info("Calculated stress level: %s", combined_score)
        return round(combined_score, 2)
    except Exception as e:
        logger.error("Stress level calculation failed: %s", str(e), exc_info=True)
        raise