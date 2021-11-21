try:
    from fraud_detector.prediction import predict
except ModuleNotFoundError:
    from prediction import predict
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI()


class Transaction(BaseModel):
    id: str
    a: Optional[int]
    b: Optional[float]
    c: Optional[float]
    d: Optional[float]
    e: Optional[float]
    f: Optional[float]
    g: Optional[str]
    h: Optional[int]
    i: Optional[str]
    j: Optional[str]
    k: Optional[float]
    l: Optional[float]
    m: Optional[float]
    n: Optional[int]
    o: Optional[str]
    p: Optional[str]
    fecha: Optional[str]
    monto: Optional[float]


@app.get("/")
def home():
    return "O detector de fraudes está funcionando corretamente!"


@app.post("/predict")
def prediction(transaction: Transaction):
    return predict(transaction.dict())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, debug=True)
