from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class StressEmotionPrediction(BaseModel):
    predicted_stress: dict  # e.g., {"label": "High"}
    predicted_emotion: dict  # e.g., {"label": "Depressed"}