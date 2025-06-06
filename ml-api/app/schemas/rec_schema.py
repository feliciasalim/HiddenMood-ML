from pydantic import BaseModel

class RecInput(BaseModel):
    text: str
    top_n: int = 2

class RecOutput(BaseModel):
    content: str
    Link: str
    Similarity: float
