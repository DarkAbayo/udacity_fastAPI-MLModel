# Script to compute model performance on data slices.

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
    compute_slice_metrics,
    inference,
    train_model
)

# Load the data
data_path = project_root / "data" / "census.csv"
data = pd.read_csv(data_path)

# Clean the data
data = clean_data(data)

# Split data
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

# Process data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary",
    training=False, encoder=encoder, lb=lb
)

# Load or train model
model_path = project_root / "model.pkl"
if model_path.exists():
    model = joblib.load(model_path)
else:
    model = train_model(X_train, y_train)

# Make predictions on test set
predictions = inference(model, X_test)

# Convert predictions back to original labels for slicing
# We need the original test dataframe for slicing
test_original = test.copy()

# Compute slice metrics for 'education' feature
feature_to_slice = "education"
slice_metrics = compute_slice_metrics(
    test_original, feature_to_slice, y_test, predictions
)

# Write results to file
output_file = project_root / "slice_output.txt"
with open(output_file, 'w') as f:
    f.write(f"Model Performance on Slices of '{feature_to_slice}'\n")
    f.write("=" * 60 + "\n\n")

    for value, metrics in sorted(slice_metrics.items()):
        f.write(f"Feature Value: {value}\n")
        f.write(f"  Number of Samples: {metrics['n_samples']}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall: {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score: {metrics['fbeta']:.4f}\n")
        f.write("\n")

print(f"Slice metrics written to {output_file}")
