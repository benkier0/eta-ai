from typing import Union
from train import train_model
from fastapi import FastAPI
from train import predict

app = FastAPI()

model = train_model()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict/")
def read_item(distance: Union[int, None] = 0, weight: Union[int, None] = 0):
    prediction = predict(model, [distance, weight])
    return {"prediction": prediction[0]}