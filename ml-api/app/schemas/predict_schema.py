from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class StressEmotionPrediction(BaseModel):
    predicted_stress: dict 
    predicted_emotion: dict 