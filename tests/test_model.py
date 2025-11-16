import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from starter.ml.model import train_model, inference, compute_model_metrics
from starter.ml.data import process_data
from starter.ml.data import clean_data

# Load the data
data_path = project_root / "data" / "census.csv"
data = pd.read_csv(data_path)

# clean the data
data = clean_data(data)

# Create test data
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Daten verarbeiten
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", 
    training=False, encoder=encoder, lb=lb
)

def test_train_model():
    # Test if train_model() returns a model
    model = train_model(X_train, y_train)  # Processed data!
    
    # Check type
    assert model is not None
    
    assert isinstance(model, RandomForestClassifier)
    # Check if model is trained (has .predict method)
    assert hasattr(model, 'predict')

def test_inference():
    # Train the model
    model = train_model(X_train, y_train)
    
    # Test inference
    predictions = inference(model, X_test)
    
    # Check type and shape
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_test)
    assert predictions.dtype in [np.int64, np.int32, np.int8]

def test_compute_model_metrics():
    # Train the model
    model = train_model(X_train, y_train)
    
    # Make predictions
    predictions = inference(model, X_test)
    
    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    
    # Check type and value range
    assert isinstance(precision, (float, np.floating))
    assert isinstance(recall, (float, np.floating))
    assert isinstance(fbeta, (float, np.floating))
    
    # Values should be between 0 and 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1