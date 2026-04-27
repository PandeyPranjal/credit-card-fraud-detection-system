from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Professional Input Schema
# -----------------------------
class FraudInput(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Time: float

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(input: FraudInput, threshold: float = 0.5):

    # Convert input to array
    data = np.array([
        input.V1, input.V2, input.V3, input.V4, input.V5,
        input.V6, input.V7, input.V8, input.V9, input.V10,
        input.V11, input.V12, input.V13, input.V14, input.V15,
        input.V16, input.V17, input.V18, input.V19, input.V20,
        input.V21, input.V22, input.V23, input.V24, input.V25,
        input.V26, input.V27, input.V28,
        input.Amount, input.Time
    ]).reshape(1, -1)

    # Get probability
    probability = model.predict_proba(data)[0][1]

    # Apply custom threshold
    prediction = int(probability > threshold)

    return {
        "prediction": prediction,
        "fraud_probability": float(probability),
        "threshold_used": threshold
    }
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)