# Part 2 – Supervised Machine Learning

## Objective
Build and evaluate regression and classification models using the cleaned dataset from Part 1.

## Dataset
- Input: `cleaned_data.csv`
- Regression Target: `body_length`
- Classification Target: `label` (0 = Legitimate, 1 = Phishing)

## Preprocessing
- Created `sender_domain` from the sender email.
- Applied One-Hot Encoding to the categorical feature (`sender_domain`) with `drop_first=True`.
- Split the dataset into 80% training and 20% testing.
- Applied `StandardScaler` by fitting only on the training data to prevent data leakage.

## Regression Models
- Linear Regression
- Ridge Regression (`alpha=1.0`)

**Evaluation Metrics**
- Mean Squared Error (MSE)
- R² Score
- Feature coefficients
- Top 3 important features

## Classification Model
- Logistic Regression (`max_iter=1000`)
- Used `class_weight="balanced"` if class imbalance was detected.

**Evaluation Metrics**
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curve
- AUC Score

## Threshold Sensitivity
Evaluated thresholds from **0.30 to 0.70** and compared:
- Precision
- Recall
- F1-Score

The threshold with the highest F1-score was identified.

## Regularization
Compared:
- Logistic Regression (`C=1.0`)
- Logistic Regression (`C=0.01`)

Compared Precision, Recall and AUC.

## Bootstrap Analysis
Performed **500 bootstrap samples** to estimate the confidence interval of the AUC difference between the two logistic regression models.

## Output Files
- `linear_regression.pkl`
- `ridge_regression.pkl`
- `logistic_regression.pkl`
- `logistic_regression_C001.pkl`
- `scaler.pkl`
- `roc_curve.png`
- `regression_comparison.csv`
- `regularization_comparison.csv`
- `threshold_sensitivity.csv`