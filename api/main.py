import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(title="Customer Segmentation API", version="1.0")

# Load pipeline once at startup
with open("customer_segmentation_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

class RFMIn(BaseModel):
    recency: float = Field(..., ge=0)
    frequency: float = Field(..., ge=0)
    monetary_value: float = Field(..., ge=0)

class PredOut(BaseModel):
    cluster: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredOut)
def predict(inp: RFMIn):
    X = np.array([[inp.recency, inp.frequency, inp.monetary_value]], dtype=float)
    # pipeline = scaler -> kmeans
    cluster = int(pipeline.predict(X)[0])
    return {"cluster": cluster}


class RFMRow(BaseModel):
    recency: float = Field(..., ge=0)
    frequency: float = Field(..., ge=0)
    monetary_value: float = Field(..., ge=0)

class BatchPredOut(BaseModel):
    clusters: List[int]

@app.post("/predict_batch", response_model=BatchPredOut)
def predict_batch(rows: List[RFMRow]):
    X = np.array([[r.recency, r.frequency, r.monetary_value] for r in rows], dtype=float)
    clusters = pipeline.predict(X).astype(int).tolist()
    return {"clusters": clusters}