from pydantic import BaseModel

class TextInput(BaseModel):
    text: str
    stress_level: float
    emotion: str

class GenAIOutput(BaseModel):
    analysis: str