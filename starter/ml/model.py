from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall,
    and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    X_pred = model.predict(X)
    return X_pred


def compute_slice_metrics(df, feature, y, preds):
    """
    Compute model performance metrics for each unique value of a feature.

    This function calculates precision, recall, and F1 score for each slice
    of data where a categorical feature has a specific value.

    Inputs
    ------
    df : pd.DataFrame
        Original dataframe with the feature column.
    feature : str
        Name of the categorical feature to slice on.
    y : np.ndarray
        True labels (binarized).
    preds : np.ndarray
        Predicted labels (binarized).
    Returns
    -------
    slice_metrics : dict
        Dictionary with feature values as keys and metrics as values.
        Format: {value: {'precision': float, 'recall': float, 'fbeta': float}}
    """
    slice_metrics = {}
    unique_values = df[feature].unique()

    for value in unique_values:
        # Get indices where feature equals this value
        mask = df[feature] == value
        if mask.sum() > 0:  # Only compute if there are samples
            y_slice = y[mask]
            preds_slice = preds[mask]
            precision, recall, fbeta = compute_model_metrics(
                y_slice, preds_slice
            )
            slice_metrics[value] = {
                'precision': precision,
                'recall': recall,
                'fbeta': fbeta,
                'n_samples': mask.sum()
            }

    return slice_metrics
