import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import train_test_split

# Add project root to Python path
# This must be done after standard library imports but before local imports
# because the local 'starter' module is not in the default Python path when
# running tests. We need to modify sys.path before importing from
# 'starter.ml.*'
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Local imports must come after sys.path modification
# noqa: E402 is used because flake8 expects all imports at the top,
# but we need sys.path.insert() to run first for these imports to work
from starter.ml.data import clean_data, process_data  # noqa: E402
from starter.ml.model import (  # noqa: E402
    compute_model_metrics,
    inference,
    train_model
)

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
    """
    Test that train_model() returns a trained RandomForestClassifier.

    Verifies that:
    - Function returns a model object
    - Model is of correct type (RandomForestClassifier)
    - Model has been trained (has .predict method)
    """
    model = train_model(X_train, y_train)

    # Check type
    assert model is not None
    assert isinstance(model, RandomForestClassifier)
    # Check if model is trained (has .predict method)
    assert hasattr(model, 'predict')


def test_inference():
    """
    Test that inference() returns correct predictions.

    Verifies that:
    - Function returns numpy array
    - Predictions have correct shape (match input)
    - Predictions are integers (0 or 1)
    """
    # Train the model
    model = train_model(X_train, y_train)

    # Test inference
    predictions = inference(model, X_test)

    # Check type and shape
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_test)
    assert predictions.dtype in [np.int64, np.int32, np.int8]


def test_compute_model_metrics():
    """
    Test that compute_model_metrics() returns valid metrics.

    Verifies that:
    - Function returns three float values (precision, recall, fbeta)
    - All metrics are between 0 and 1 (valid range)
    """
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
