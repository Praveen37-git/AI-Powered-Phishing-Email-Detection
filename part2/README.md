# Part 2 – Supervised Machine Learning

## Dataset

The cleaned dataset (`cleaned_data.csv`) from Part 1 was used. It contains 11,054 records with numeric features. The classification target is `class` (`-1` = Legitimate, `1` = Phishing). For regression, `DomainRegLen` was used as the continuous target.

## Features and Preprocessing

* **Feature Matrix (X):** All columns except `DomainRegLen` and `class`.
* **Regression Target (y_reg):** `DomainRegLen`
* **Classification Target (y_clf):** `class` (converted to 0/1).

The dataset contains only numeric columns, so no label encoding or one-hot encoding was required.

The data was split into 80% training and 20% testing sets using `random_state=42`. `StandardScaler` was fitted only on the training data to prevent data leakage.

## Regression

Linear Regression and Ridge Regression (`alpha=1.0`) were trained and evaluated using Mean Squared Error (MSE) and R² score. Model coefficients were examined, and the three features with the largest absolute coefficients were identified. Ridge Regression was compared with Linear Regression to observe the effect of L2 regularization.

## Classification

Logistic Regression (`max_iter=1000`) was trained to classify phishing websites. Class imbalance was checked and handled if necessary. Model performance was evaluated using a confusion matrix, accuracy, precision, recall, F1-score, ROC curve, and AUC.

Decision thresholds from **0.30 to 0.70** were tested to compare precision, recall, and F1-score. A second Logistic Regression model (`C=0.01`) was trained to study the effect of stronger regularization.

## Bootstrap Analysis

A bootstrap experiment with **500 samples** was performed to estimate the 95% confidence interval of the AUC difference between the two Logistic Regression models. The results were used to determine whether the performance difference was statistically reliable.
