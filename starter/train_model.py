"""
Script to train machine learning model.

This script:
1. Loads and cleans the census.csv data
2. Splits data into training and test sets
3. Processes data using one-hot encoding and label binarization
4. Trains a Random Forest Classifier
5. Saves the trained model, encoder, and label binarizer
6. Evaluates the model and prints performance metrics
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to Python path so imports work
# This must be done after standard library imports but before local imports
# because the local 'starter' module is not in the default Python path.
# We need to modify sys.path before importing from 'starter.ml.*'
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

# Optional enhancement, use K-fold cross validation instead of
# a train-test split.
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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary",
    training=False, encoder=encoder, lb=lb
)

# Train the model
model = train_model(X_train, y_train)

# Save the model
joblib.dump(model, project_root / "model.pkl")
joblib.dump(encoder, project_root / "encoder.pkl")
joblib.dump(lb, project_root / "lb.pkl")

# Compute the model metrics
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Fbeta: {fbeta}")
