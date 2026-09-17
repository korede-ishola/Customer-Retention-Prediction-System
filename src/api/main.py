from fastapi import FastAPI
from src.serving.inference import make_prediction
from src.api.app import build_app

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Telco Churn Prediction API"}

@app.post("/predict")
def predict(data: dict):
    prediction, _ = make_prediction(data)
    return prediction

app = build_app(app, path='/ui')