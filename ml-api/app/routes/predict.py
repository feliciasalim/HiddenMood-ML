from fastapi import APIRouter
from app.schemas.predict_schema import TextInput, StressEmotionPrediction
from app.schemas.rec_schema import VideoRecommendationOutput
from app.schemas.genai_schema import GenAIOutput
from app.services.predict_model import predict_stress_emotion, calculate_stress_level, preprocess_text
from app.services.rec_system import recommend_video
from app.services.geminiAi import analyze_with_vertex
import sqlite3
from datetime import datetime

router = APIRouter()

conn = sqlite3.connect("feedback.db", check_same_thread=False)
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
    text = data.text
    vectorized_input = preprocess_text(text)

    prediction = predict_stress_emotion(text)
    stress_label = prediction["predicted_stress"]["label"]
    emotion_label = prediction["predicted_emotion"]["label"]
    stress_logits = prediction["stress_logits"]

    stress_level = calculate_stress_level(text, vectorized_input, stress_logits)

    recommended_videos = recommend_video(emotion_label)

    analysis = analyze_with_vertex(stress_level, text, emotion_label)

    conn.execute("""
        INSERT INTO feedback (user_id, text, stress_label, stress_level, emotion_label, date)
        VALUES (?, ?, ?, ?, ?, ?)
    """, ("user123", text, stress_label, stress_level, emotion_label, datetime.utcnow().isoformat()))
    conn.commit()

    return {
        "predicted_stress": prediction["predicted_stress"],
        "predicted_emotion": prediction["predicted_emotion"],
        "stress_level": {"stress_level": stress_level},
        "recommended_videos": {"recommendations": recommended_videos},
        "analysis": analysis
    }

@router.get("/feedback/history")
def get_feedback_history():
    cursor = conn.execute("SELECT text, stress_level, date FROM feedback WHERE user_id = ?", ("user123",))
    feedbacks = [{"text": row[0], "percentage": row[1], "date": row[2]} for row in cursor.fetchall()]
    return feedbacks
