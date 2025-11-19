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
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import pandas as pd
from starter.ml.model import inference
from starter.ml.data import process_data
from starter.ml.data import clean_data

# Initialize FastAPI app instance
app = FastAPI()

# Get the directory where main.py is located
# This ensures the .pkl files are found regardless of working directory
BASE_DIR = Path(__file__).parent

# Load model, encoder, and label binarizer at startup
model = joblib.load(BASE_DIR / "model.pkl")
encoder = joblib.load(BASE_DIR / "encoder.pkl")
lb = joblib.load(BASE_DIR / "lb.pkl")

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
    """
    Pydantic model for Census income prediction API request.

    This model validates the input data for the POST /inference endpoint.
    Field aliases are used to handle column names with hyphens (e.g.,
    "education-num" maps to education_num in Python).

    Attributes:
        age: Age of the individual
        workclass: Type of employment
        fnlgt: Final weight (demographic weighting)
        education: Highest level of education
        education_num: Numeric representation of education
        marital_status: Marital status
        occupation: Type of occupation
        relationship: Relationship status
        race: Race
        sex: Gender
        capital_gain: Capital gains
        capital_loss: Capital losses
        hours_per_week: Hours worked per week
        native_country: Country of origin
    """
    age: int = Field(..., example=39)
    workclass: str = Field(..., alias="workclass", example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(
        ..., alias="marital-status", example="Never-married"
    )
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(
        ..., alias="native-country", example="United-States"
    )


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

    Args:
        data: CensusData model containing all required features

    Returns:
        dict: Prediction result with income category
            Example: {"prediction": ">50K"} or {"prediction": "<=50K"}
    """

    # Convert Pydantic model to DataFrame
    # get a dictionary with the alias names
    data_dict = data.model_dump(by_alias=True)
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
