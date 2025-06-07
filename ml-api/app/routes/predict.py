import logging
from fastapi import APIRouter, HTTPException
from app.schemas.predict_schema import TextInput, StressEmotionPrediction
from app.schemas.rec_schema import VideoRecommendationOutput
from app.schemas.genai_schema import GenAIOutput
from app.services.predict_model import predict_stress_emotion, calculate_stress_level, preprocess_text
from app.services.rec_system import recommend_video
from app.services.geminiAi import analyze_with_vertex
import sqlite3
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

conn = sqlite3.connect("/tmp/feedback.db", check_same_thread=False)
conn.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        text TEXT,
        stress_label TEXT,
        stress_level FLOAT,
        emotion_label TEXT,
        date TEXT
    )
""")
conn.commit()

@router.post("/analyze", response_model=dict)
def analyze_all(data: TextInput):
    try:
        logger.info("Received request with text: %s", data.text)
        text = data.text
        
        logger.info("Preprocessing text")
        vectorized_input = preprocess_text(text)
        
        logger.info("Predicting stress and emotion")
        prediction = predict_stress_emotion(text)
        stress_label = prediction["predicted_stress"]["label"]
        emotion_label = prediction["predicted_emotion"]["label"]
        stress_logits = prediction["stress_logits"]
        
        logger.info("Calculating stress level")
        stress_level = calculate_stress_level(text, vectorized_input, stress_logits)
        
        logger.info("Recommending videos")
        recommended_videos = recommend_video(emotion_label)
        
        logger.info("Analyzing with Vertex AI")
        analysis = analyze_with_vertex(stress_level, text, emotion_label)
        
        logger.info("Inserting feedback into database")
        conn.execute("""
            INSERT INTO feedback (user_id, text, stress_label, stress_level, emotion_label, date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("user123", text, stress_label, stress_level, emotion_label, datetime.utcnow().isoformat()))
        conn.commit()
        
        logger.info("Returning response")
        return {
            "predicted_stress": prediction["predicted_stress"],
            "predicted_emotion": prediction["predicted_emotion"],
            "stress_level": {"stress_level": stress_level},
            "recommended_videos": {"recommendations": recommended_videos},
            "analysis": analysis
        }
    except Exception as e:
        logger.error("Error in /analyze endpoint: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")