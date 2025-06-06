from fastapi import APIRouter
from app.schemas.rec_schema import RecInput, RecOutput
from app.services.rec_system import recommend_video

router = APIRouter()

@router.post("/recommend", response_model=list[RecOutput])
def get_recommendation(data: RecInput):
    return recommend_video(data.text, top_n=data.top_n)
