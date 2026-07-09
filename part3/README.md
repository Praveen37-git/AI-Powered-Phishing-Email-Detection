# Part 3 – Advanced Modeling

## Decision Tree

### Baseline Decision Tree

* Training Accuracy: **1.0000**
* Test Accuracy: **0.9765**

The baseline decision tree shows signs of **overfitting** because it achieves perfect training accuracy but lower test accuracy. Decision trees are high-variance models since they greedily choose the best split at each node without revisiting previous decisions.

### Controlled Decision Tree

* Parameters:

  * `max_depth = 5`
  * `min_samples_split = 20`
* Training Accuracy: **0.9572**
* Test Accuracy: **0.9535**

`max_depth` limits the depth of the tree to reduce overfitting, while `min_samples_split` prevents splitting nodes with too few samples. The controlled tree has a smaller train-test gap, indicating better generalization.

---

## Gini vs Entropy

**Gini Impurity**
Gini = 1 − Σpi²

Entropy = −Σpi log₂(pi)

A node with **Gini = 0** is completely pure, meaning all samples belong to a single class.

---

## Random Forest

* Training Accuracy: 0.9791
* Test Accuracy: 0.9739
* ROC-AUC: 0.9965

Top 5 important features were identified using `feature_importances_`.

Random Forest feature importance measures the average reduction in Gini impurity contributed by a feature across all trees. Unlike linear regression coefficients, feature importance does not indicate positive or negative relationships.

### Bagging

Random Forest uses **bootstrap sampling** to train each tree on a random sample of the training data. At every split, only a random subset of features is considered. Combining many trees reduces variance and improves generalization compared to a single decision tree.

---

## Gradient Boosting

* Training Accuracy: 0.9732
* Test Accuracy: 0.9696
* ROC-AUC: 0.9954

---

## Feature Ablation

The five least important features were removed and a new Random Forest was trained.

* Full Model AUC: 0.996534
* Reduced Model AUC: 0.996771

If the AUC remains similar, the removed features contribute little and can be excluded to simplify the model.

---

## Cross Validation

5-fold Stratified Cross Validation was performed using ROC-AUC.

| Model                    |   Mean AUC | Std AUC |
| ------------------------ | ---------: | ------: |
| Logistic Regression      |     0.9547 |  0.0023 |
| Controlled Decision Tree |     0.9796 |  0.0006 |
| Random Forest            |     0.9967 |  0.0006 |
| Gradient Boosting        |     0.9955 |  0.0006 |


Cross-validation provides a more reliable estimate of model performance because every sample is used for both training and validation across different folds.

---

## Grid Search

The Random Forest was tuned using GridSearchCV.

* Best Parameters:

  * `n_estimators = 200`
  * `max_depth = None`
  * `min_samples_leaf = 1`
* Best 5-fold CV AUC: **0.9985**

A total of **18 parameter combinations** were evaluated across **5 folds**, resulting in **90 model fits**. Grid Search exhaustively evaluates all parameter combinations, whereas Randomized Search tests only a random subset, making it faster but less exhaustive.

---

## Learning Curve

Training fractions of **20%, 40%, 60%, 80%, and 100%** were evaluated.

Training AUC decreased slightly as more data was used, while test AUC increased and stabilized. This indicates improved generalization with additional training data.

---

## Model Serialization

The best pipeline was saved as **`best_model.pkl`** using `joblib.dump()` and successfully reloaded using `joblib.load()` to make predictions on new samples.

---

## Final Model Comparison

| Model               |        Test AUC |
| ------------------- | --------------: |
| Logistic Regression |          0.9547 |
| Decision Tree       |          0.9796 |
| Random Forest       |          0.9965 |
| Gradient Boosting   |          0.9954 |

### Recommendation

Random Forest is the recommended model because it achieved the highest ROC-AUC and showed strong generalization on unseen data. Its ensemble approach reduces overfitting and provides the most reliable performance for phishing email detection.