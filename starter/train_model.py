# Script to train machine learning model.

import sys
from pathlib import Path

# Add project root to Python path so imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import train_model
from starter.ml.data import clean_data
from starter.ml.model import compute_model_metrics, inference

import pandas as pd
import joblib

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
