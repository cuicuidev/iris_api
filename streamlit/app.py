import streamlit as st
# import pickle as pkl
# import os

import requests
from pydantic import BaseModel

class Request(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

def main():
    st.title("Iris Species Prediction")
    sepal_length = st.slider("Sepal Length", 0.0, 10.0, 5.0)
    sepal_width = st.slider("Sepal Width", 0.0, 10.0, 5.0)
    petal_length = st.slider("Petal Length", 0.0, 10.0, 5.0)
    petal_width = st.slider("Petal Width", 0.0, 10.0, 5.0)

    # features = {
    #     "sepal_length": sepal_length,
    #     "sepal_width": sepal_width,
    #     "petal_length": petal_length,
    #     "petal_width": petal_width
    # }

    features = Request(
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width
    )

    if st.button("Predict"):
        predict(features)


def predict(features):
    # model = load_model()
    # encoder = load_encoder()
    # input_data = list(features.values())
    # prediction = model.predict([input_data])
    # prediction = encoder.inverse_transform(prediction)[0]
    # st.success(f"The iris species is {prediction}")

    endpoint = "http://localhost:8000/predict"

    response = requests.post(endpoint, json=features.model_dump())
    result = response.json()
    result = result["prediction"]

    st.success(f"The iris species is {result}")

# def load_model():
#     with open("model.pkl", "rb") as f:
#         model = pkl.load(f)
#     return model

# def load_encoder():
#     with open("encoder.pkl", "rb") as f:
#         encoder = pkl.load(f)
#     return encoder

if __name__ == "__main__":
    main()
