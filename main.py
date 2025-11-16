"""
FastAPI application for Census Income Prediction Model.

This module implements a RESTful API with:
- GET endpoint: Welcome message
- POST endpoint: Model inference for income prediction
"""

# TODO: Add necessary imports here
# Hint: You'll need FastAPI, Pydantic, and your ML model functions
# from fastapi import FastAPI
# from pydantic import BaseModel, Field
# import joblib or pickle to load the model
# from starter.ml.model import inference
# from starter.ml.data import process_data

from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from starter.ml.model import inference
from starter.ml.data import process_data
from starter.ml.data import clean_data

# Initialize FastAPI app instance
app = FastAPI()

# Load model, encoder, and label binarizer at startup
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
lb = joblib.load("lb.pkl")

# Define categorical features (same as in train_model.py)
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

class CencusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., alias="workclass", example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

@app.get("/")
def welcome() -> dict:
    """
    Welcome endpoint that returns a greeting message.
    """
    return {"message": "Welcome to the Census Income Prediction API"}

@app.post("/inference")
def predict_income(data: CencusData) -> dict:
    """
    Predict income category based on census data.
    #     
    #     Args:
    #         data: CensusData model containing all required features
    #     
    #     Returns:
    #         dict: Prediction result with income category
    #             Example: {"prediction": ">50K"} or {"prediction": "<=50K"}
    """

    # Convert Pydantic model to DataFrame
    data_dict = data.model_dump(by_alias=True) # get a dictionary with the alias names
    df = pd.DataFrame([data_dict])

    # Clean the data
    df = clean_data(df)

    # Process the data using process_data() function
    X_processed, _, _, _ = process_data(
        df, 
        categorical_features=CATEGORICAL_FEATURES, 
        encoder=encoder, 
        lb=lb, 
        training=False
    )

    # Run inference using the inference() function
    prediction = inference(model, X_processed)

    # Convert prediction to human-readable format
    prediction_label = lb.inverse_transform(prediction)[0]

    return {"prediction": prediction_label}



