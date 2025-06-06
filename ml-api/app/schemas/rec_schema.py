from pydantic import BaseModel
from typing import List

class TextInput(BaseModel):
    text: str

class VideoRecommendationOutput(BaseModel):
    recommendations: List[str] 