from fastapi import FastAPI
from app.routes.predict import router as predict_router

app = FastAPI()

app.include_router(predict_router, prefix="/predict")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)