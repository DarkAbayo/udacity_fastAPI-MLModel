# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- **Model Name**: Census Income Prediction Model
- **Model Type**: Random Forest Classifier
- **Framework**: scikit-learn
- **Version**: 1.0
- **Date Created**: 2024
- **Developer**: ML DevOps Project

The model is a Random Forest Classifier with 100 estimators, trained to predict whether an individual's income exceeds $50,000 per year based on census demographic data.

## Intended Use

This model is intended for educational and demonstration purposes in the context of ML DevOps practices. It demonstrates how to deploy a machine learning model as a RESTful API using FastAPI.

**Primary Use Case**: Predict income category (≤50K or >50K) based on demographic features from census data.

**Out-of-Scope Use Cases**: 
- This model should not be used for actual financial decisions or policy making
- The model is trained on historical data and may not reflect current economic conditions
- The model should not be used for discriminatory purposes

## Training Data

- **Dataset**: UCI Census Income Dataset
- **Source**: https://archive.ics.uci.edu/ml/datasets/census+income
- **Training Set Size**: 80% of total data (approximately 26,000 samples)
- **Features**: 
  - Numerical: age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Target Variable**: salary (≤50K or >50K)
- **Data Preprocessing**: 
  - Whitespace removed from column names and string values
  - One-Hot Encoding applied to categorical features
  - Label Binarization applied to target variable
- **Train-Test Split**: 80/20 split with random_state=42 for reproducibility

## Evaluation Data

- **Test Set Size**: 20% of total data (approximately 6,500 samples)
- **Evaluation Method**: Hold-out validation set
- **Data Distribution**: Same preprocessing as training data, using the same encoders and label binarizer

## Metrics

The model performance is evaluated using the following metrics:

- **Precision**: Measures the accuracy of positive predictions (proportion of predicted >50K that are actually >50K)
- **Recall**: Measures the ability to find all positive instances (proportion of actual >50K that are correctly identified)
- **F1-Score**: Harmonic mean of precision and recall (beta=1)

**Overall Model Performance on Test Set**:
- Precision: 0.7419 (74.19%)
- Recall: 0.6384 (63.84%)
- F1-Score: 0.6863 (68.63%)

**Slice Performance**: The model performance varies across different slices of the data. For example, when slicing by education level, the model shows different performance metrics for each education category. Detailed slice metrics are available in `slice_output.txt`.

The model demonstrates reasonable performance but shows some bias across different demographic groups, which is an important consideration for ethical deployment.

## Ethical Considerations

**Bias and Fairness**: 
- The model may exhibit different performance across different demographic groups (race, sex, education, etc.)
- Slice analysis reveals performance disparities that should be addressed before production deployment
- The model should not be used to make decisions that could discriminate against protected groups

**Privacy**: 
- The model uses demographic data that could potentially be used to identify individuals
- Care should be taken when deploying this model to ensure compliance with privacy regulations

**Transparency**: 
- This model card provides transparency about model performance and limitations
- Users should be aware of the model's limitations and potential biases

**Recommendations**:
- Conduct thorough bias analysis before production deployment
- Consider fairness constraints or post-processing techniques to reduce bias
- Regularly monitor model performance across different demographic groups
- Retrain the model periodically with updated data to reflect current conditions

## Caveats and Recommendations

**Limitations**:
1. The model is trained on historical data from 1994 and may not reflect current economic conditions
2. The model shows performance disparities across different demographic groups
3. The dataset may contain biases that are reflected in the model predictions
4. The model is designed for binary classification and does not provide probability estimates or confidence intervals

**Recommendations for Use**:
1. **Do not use for critical financial decisions**: This is a demonstration model and should not be used for actual financial or policy decisions
2. **Monitor performance**: Regularly evaluate model performance on new data and across different demographic groups
3. **Address bias**: Implement bias mitigation techniques before production deployment
4. **Update regularly**: Retrain the model with fresh data to maintain relevance
5. **Provide transparency**: Always disclose model limitations and potential biases to end users

**Technical Recommendations**:
- Consider using cross-validation for more robust performance estimates
- Experiment with different hyperparameters to improve performance
- Implement feature importance analysis to understand model decisions
- Add model versioning and monitoring in production

**Deployment Considerations**:
- The model requires preprocessing (data cleaning and encoding) before inference
- Ensure consistent preprocessing between training and inference
- Monitor model drift and retrain when performance degrades
- Implement proper error handling and logging in production

