import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, accuracy_score,f1_score,precision_score,recall_score
import matplotlib.pyplot as plt

#Get the cleaned dataset  from part1
FILE_PATH = "part2/data/cleaned_data.csv"
OUTPUT_DIR = "part2/output"

def load_data():
    #Using try except blocks for error handling, if there's any issues with the filepath
    try:
        df = pd.read_csv(FILE_PATH)
        print("Dataset loaded successfully")
        #print number of rows and columns in the dataset and return the dataframe
        print(f"No. of Rows: {df.shape[0]}")
        print(f"No. of Columns: {df.shape[1]}")
        return df
    except FileNotFoundError:
        print("Dataset not found, check the file path")
        return None
    
#Exploring the dataset
def explore_data(df):
    print("First 5 rows:")
    print(df.head())
    print("\nDataset information: ")
    df.info()
    print("\nShape of the dataset: ")
    print(df.shape)
    print("\nColumn names: ")
    for column in df.columns:
        print(f"- {column}")

def feature_target_definition(df):
    """
    Define feature matrix and target variables
    create categorical feature email domain
    """  
    df["sender_domain"] = (df["sender"].str.extract(r'@([^>]+)',expand=False).fillna("unknown"))  
    top_domains = df["sender_domain"].value_counts().nlargest(20).index
    df["sender_domain"] = df["sender_domain"].where(
        df["sender_domain"].isin(top_domains),
        "other"
    )
    X = df[[
        "sender_domain",
        "urls",
        "subject_length",
        "word_count",
        "uppercase_count",
        "special_characters",
        "question_marks",
        "exclamation_marks",
        "contains_http"
    ]]
    #Regression target column
    y_reg = df["body_length"]
    #classification target column
    y_clf = df["label"]
    return X,y_reg,y_clf

def split_data(X,y_reg,y_clf):
    """
    Splitting the dataset into train and test sets for 
    both regression and classification models
    """
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(X,y_reg,y_clf,test_size=0.2,random_state=42,stratify=y_clf)

    return X_train,X_test,y_train_reg,y_test_reg,y_train_clf,y_test_clf

def scale_data(X_train,X_test):
    """
    Scale the dataset using Standard Scaler
    fit the scaler only on training features
    save the scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
    return X_train_scaled, X_test_scaled

def check_class_balance(y_train_clf):
    print("\n===== Class Distribution =====")
    class_percentage = y_train_clf.value_counts(normalize=True) * 100
    print(class_percentage)
    return class_percentage
        

def linear_regression_model(X_train_scaled, X_test_scaled, y_train_reg, y_test_reg, feature_names):
    """
    Initialize the model and train the model using scaled features and 
    generate predictions on the test set
    """
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train_reg)
    y_pred_reg = lr_model.predict(X_test_scaled)
    lr_mse = mean_squared_error(y_test_reg, y_pred_reg)
    lr_r2 = r2_score(y_test_reg, y_pred_reg)
    print("\n===== Linear Regression Performance =====")
    print(f"Mean squared error: {lr_mse:.4f}")
    print(f"R² score: {lr_r2:.4f}")
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": lr_model.coef_})
    coef_df["Absolute_value"] = coef_df["Coefficient"].abs()
    top_features = coef_df.sort_values(by="Absolute_value", ascending=False).head(3)
    print("\n===== Linear coefficients =====")
    print(coef_df[["Feature","Coefficient"]].to_string(index=False))
    print("\n===== Top 3 features (Absolute coefficients) =====")
    print(top_features[["Feature","Coefficient"]].to_string(index=False))
    return lr_model, lr_mse, lr_r2

def ridge_regression_model(X_train_scaled, X_test_scaled, y_train_reg, y_test_reg, feature_names):
    """
    Train and evaluate Ridge Regression using alpha=1.0.
    Prints MSE, R² score and feature coefficients.
    """
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train_reg)
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    ridge_mse = mean_squared_error(y_test_reg, y_pred_ridge)
    ridge_r2 = r2_score(y_test_reg,y_pred_ridge)
    print("\n===== Ridge Regression Performance =====")
    print(f"Mean squared error: {ridge_mse:.4f}")
    print(f"R² Score: {ridge_r2:.4f}\n")
    ridge_coef_df = pd.DataFrame({"Feature": feature_names, "Ridge_Coefficient": ridge_model.coef_})
    print("\n===== Ridge Coefficients =====")
    print(ridge_coef_df.to_string(index=False))
    return ridge_model, ridge_mse, ridge_r2

def logistic_regression_model(X_train_scaled, X_test_scaled, y_train_clf, y_test_clf, class_weight):
    """
    Train the Logistic Regression model with classification samples
    Calculate the metric and plot the ROC curve
    """
    log_reg = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train_clf)
    y_pred = log_reg.predict(X_test_scaled)
    y_prob = log_reg.predict_proba(X_test_scaled)
    y_prob_positive = y_prob[:,1]
    acc_score = accuracy_score(y_test_clf,y_pred)
    prec_score = precision_score(y_test_clf,y_pred)
    recall = recall_score(y_test_clf,y_pred)
    print("\n===== Classification evaluation metrics =====")
    print("===== Confusion Matrix =====")
    print(confusion_matrix(y_test_clf, y_pred))
    print(f"Accuracy: {acc_score:.4f}")
    print(f"Precision: {prec_score:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1_score(y_test_clf,y_pred):.4f}")
    print("===== Classification Report =====")
    print(classification_report(y_test_clf,y_pred))
    print("===== Area under curve (AUC) =====")
    auc_score = roc_auc_score(y_test_clf, y_prob_positive)
    print(f"ROC-AUC score: {auc_score:.4f}")
    fpr,tpr,thresholds = roc_curve(y_test_clf,y_prob_positive)
    plt.figure(figsize=(8,5))
    plt.plot(fpr,tpr,color="blue",lw=2,label=f"ROC curve (AUC = {auc_score:.2f})")
    plt.plot([0,1],color="red",lw=2,linestyle="--",label="Random Guess (0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")
    plt.show()
    
    return log_reg, y_prob, y_prob_positive, auc_score, prec_score, recall

def threshold_sensitivity_analysis(y_test_clf, y_prob_positive):
    thresholds = np.arange(0.30,0.71,0.10)
    results = []
    for th in thresholds:
        y_pred = (y_prob_positive >= th).astype(int)
        precision = precision_score(y_test_clf,y_pred)
        recall = recall_score(y_test_clf,y_pred)
        f1 = f1_score(y_test_clf,y_pred)
        results.append({"Threshold": th, "Precision": precision, "Recall": recall, "F1": f1})
    results_df = pd.DataFrame(results)
    print("===== Decision-Threshold Sensitivity Table =====")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    max_f1 = results_df["F1"].idxmax()
    max_f1_row = results_df.loc[max_f1]
    print("\n===== Best Threshold details =====")
    print(f"Best Threshold: {max_f1_row['Threshold']:.2f}")
    print(f"Max F1 score: {max_f1_row['F1']:.4f}")
    return results_df

def logistic_regression_regularization(X_train_scaled, X_test_scaled, y_train_clf, y_test_clf, class_weight):
    """
    Train a second logistic regression with C=0.01
    Calculate the metrics and compare it with the baseline model
    """
    log_model = LogisticRegression(C=0.01,class_weight=class_weight,max_iter=1000,random_state=42)
    log_model.fit(X_train_scaled, y_train_clf)
    y_pred = log_model.predict(X_test_scaled)
    y_prob = log_model.predict_proba(X_test_scaled)[:,1]
    print("\n===== Classification evaluation metrics =====")
    print("===== Confusion Matrix =====")
    print(confusion_matrix(y_test_clf, y_pred))
    acc_score = accuracy_score(y_test_clf,y_pred)
    prec_score = precision_score(y_test_clf,y_pred)
    recall = recall_score(y_test_clf,y_pred)
    print(f"Accuracy: {acc_score:.4f}")
    print(f"Precision: {prec_score:.4f}")
    print(f"Recall: {recall:.4f}")
    print("===== Classification Report =====")
    print(classification_report(y_test_clf,y_pred))
    print("===== Area under curve (AUC) =====")
    auc_score = roc_auc_score(y_test_clf, y_prob)
    print(f"ROC-AUC score: {auc_score:.4f}")
    return log_model,prec_score,recall,auc_score,y_prob

def bootstrap_auc_difference(y_test_clf,baseline_prob,regularized_prob):
    auc_differences = []
    for i in range(500):
        indices = np.random.choice(len(y_test_clf),size=len(y_test_clf),replace=True)
        bootstrap_y_test_clf = y_test_clf.iloc[indices]
        bootstrap_baseline_prob = baseline_prob[indices]
        bootstrap_regularized_prob = regularized_prob[indices]
        if len(np.unique(bootstrap_y_test_clf)) < 2:
            continue
        roc_auc_baseline = roc_auc_score(bootstrap_y_test_clf, bootstrap_baseline_prob)
        roc_auc_regularized = roc_auc_score(bootstrap_y_test_clf, bootstrap_regularized_prob)
        difference = roc_auc_baseline - roc_auc_regularized
        auc_differences.append(difference)
    auc_differences = np.array(auc_differences)
    mean = np.mean(auc_differences)
    lower_bound = np.percentile(auc_differences,2.5)
    upper_bound = np.percentile(auc_differences,97.5)
    print("====== Bootstrap AUC differences ======")
    print(f"Mean AUC difference: {mean:.4f}")
    print(f"95% CI Lower bound: {lower_bound:.4f}")
    print(f"95% CI Upper bound: {upper_bound:.4f}")
    if lower_bound > 0 or upper_bound < 0:
        print("95 percent confidence interval excludes zero.")
    else:
        print("95 percent interval includes zero.")

def main():
    os.makedirs("part2/output",exist_ok=True)
    df = load_data()
    if df is None:
        return
    explore_data(df)
    X,y_reg,y_clf = feature_target_definition(df)
    #Apply one-hot encoding on the sender domain column
    X = pd.get_dummies(X,columns=["sender_domain"],drop_first=True)
    feature_names = X.columns.tolist()
    print(f"\nX: {X.shape}")
    print(f"\ny_reg: {y_reg.shape}")
    print(f"\ny_clf: {y_clf.shape}")
    print("Splitting data...")
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = split_data(X, y_reg, y_clf)

    print("Scaling data...")
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    print("Finished scaling")
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"\nTest samples: {X_test.shape[0]}")
    print(f"\nTraining features: {X_train.shape[1]}")
    print(f"\nTest features: {X_test.shape[1]}")
    print(f"\nX_train shape: {X_train_scaled.shape}")
    print(f"\nX_test shape: {X_test_scaled.shape}")
    print("\nFeatures used for training:")
    print(X.columns.tolist())
    print(f"\nScaler saved to {OUTPUT_DIR}/scaler.pkl")
    smallest_class_percentage = check_class_balance(y_train_clf).min()
    print(f"\nTotal features after encoding: {X.shape[1]}")
    print("\nFirst 20 features:")
    print(X.columns[:20].tolist())
    
    lr_model, lr_mse, lr_r2= linear_regression_model(X_train_scaled,X_test_scaled,y_train_reg,y_test_reg,feature_names)
    ridge_model, ridge_mse, ridge_r2 = ridge_regression_model(X_train_scaled,X_test_scaled,y_train_reg,y_test_reg,feature_names)
    joblib.dump(lr_model, f"{OUTPUT_DIR}/linear_regression.pkl")
    joblib.dump(ridge_model, f"{OUTPUT_DIR}/ridge_regression.pkl")
    comparison = pd.DataFrame({"Model": ["Linear Regression","Ridge Regression"], "MSE":[lr_mse,ridge_mse],"R2":[lr_r2,ridge_r2]})
    print("\n===== Linear and Ridge regression comparison =====")
    print(comparison.to_string(index=False))
    comparison.to_csv(f"{OUTPUT_DIR}/regression_comparison.csv",index=False)
    if(smallest_class_percentage <= 35):
        class_weight = "balanced"
    else:
        class_weight = None
    log_reg, y_prob, y_prob_positive, log_auc, log_precision, log_recall = logistic_regression_model(X_train_scaled,X_test_scaled,y_train_clf,y_test_clf,class_weight)
    joblib.dump(log_reg, f"{OUTPUT_DIR}/logistic_regression.pkl")
    results = threshold_sensitivity_analysis(y_test_clf, y_prob_positive)
    results.to_csv(f"{OUTPUT_DIR}/threshold_sensitivity.csv", index = False)
    reg_model,reg_precision,reg_recall,reg_auc,reg_prob_positive = logistic_regression_regularization(X_train_scaled,X_test_scaled,y_train_clf,y_test_clf,class_weight)
    comparison = pd.DataFrame({"Model": ["Logistic Regression (C=1.0)","Logistic Regression (C=0.01)"], "Precision": [log_precision, reg_precision],"Recall": [log_recall, reg_recall], "AUC": [log_auc, reg_auc]})
    print(comparison.to_string(index=False))
    comparison.to_csv(f"{OUTPUT_DIR}/regularization_comparison.csv",index=False)
    joblib.dump(reg_model, f"{OUTPUT_DIR}/logistic_regression_C001.pkl")
    bootstrap_auc_difference(y_test_clf, y_prob_positive, reg_prob_positive)
    

if __name__ == "__main__":
    main()