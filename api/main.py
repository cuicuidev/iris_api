from fastapi import FastAPI
from pydantic import BaseModel
import pickle


class Request(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class Response(BaseModel):
    prediction: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(request: Request):
    model = load_model()
    encoder = load_encoder()

    input_data = [request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]
    input_data = [input_data]

    prediction = model.predict(input_data)

    prediction = encoder.inverse_transform(prediction)

    response = Response(prediction=prediction[0])
    return response


def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def load_encoder():
    with open('encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
    return encoder
