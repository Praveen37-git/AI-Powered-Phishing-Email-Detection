import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


FILE_PATH = "part3/data/cleaned_data.csv"
OUTPUT_DIR = "part3/output"

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

def decision_tree_baseline(X_train_scaled,X_test_scaled,y_train_clf,y_test_clf):
    """
    Train a Decision tree classifier and
    Compare training and test accuracy
    """
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_scaled, y_train_clf)
    y_pred_train = dt_model.predict(X_train_scaled)
    y_pred_test = dt_model.predict(X_test_scaled)
    train_acc_score = accuracy_score(y_train_clf, y_pred_train)
    test_acc_score = accuracy_score(y_test_clf, y_pred_test)
    print("\n====== Decision tree Classifier ======")
    print(f"Training Accuracy: {train_acc_score:.4f}")
    print(f"Test Accuracy: {test_acc_score:.4f}")
    return dt_model, train_acc_score, test_acc_score

def decision_tree_controlled(X_train_scaled,X_test_scaled,y_train_clf,y_test_clf):
    """
    Train a Decision Tree classifier with max_depth = 5 and
    Compare its train and test accuracy
    """
    dt_model = DecisionTreeClassifier(max_depth=5,min_samples_split=20,random_state=42)
    dt_model.fit(X_train_scaled, y_train_clf)
    y_pred_train = dt_model.predict(X_train_scaled)
    y_pred_test = dt_model.predict(X_test_scaled)
    train_acc_score = accuracy_score(y_train_clf, y_pred_train)
    test_acc_score = accuracy_score(y_test_clf, y_pred_test)
    print("\n====== Controlled Decision tree ======")
    print(f"Training Accuracy: {train_acc_score:.4f}")
    print(f"Test Accuracy: {test_acc_score:.4f}")
    return dt_model, train_acc_score, test_acc_score

def decision_tree_gini(X_train_scaled,X_test_scaled,y_train_clf,y_test_clf):
    """
    Train a Decision Tree classifier with criterion="gini" and
    calculate it's test accuracy
    """
    dt_model = DecisionTreeClassifier(max_depth=5,min_samples_split=20,criterion="gini",random_state=42)
    dt_model.fit(X_train_scaled, y_train_clf)
    y_pred_test = dt_model.predict(X_test_scaled)
    test_acc_score = accuracy_score(y_test_clf, y_pred_test)
    print("\n====== Gini Decision tree ======")
    print(f"Test Accuracy: {test_acc_score:.4f}")
    return dt_model, test_acc_score

def decision_tree_entropy(X_train_scaled,X_test_scaled,y_train_clf,y_test_clf):
    """
    Train a Decision Tree classifier with criterion="entropy" and
    calculate it's test accuracy
    """
    dt_model = DecisionTreeClassifier(max_depth=5,min_samples_split=20,criterion="entropy",random_state=42)
    dt_model.fit(X_train_scaled, y_train_clf)
    y_pred_test = dt_model.predict(X_test_scaled)
    test_acc_score = accuracy_score(y_test_clf, y_pred_test)
    print("\n====== Entropy Decision tree ======")
    print(f"Test Accuracy: {test_acc_score:.4f}")
    return dt_model, test_acc_score

def random_forest_classifier(X_train_scaled,X_test_scaled,y_train_clf,y_test_clf, feature_names):
    """
    Train a Random Forest classifier and
    calculate training, test accuracy and ROC_AUC
    """
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train_clf)
    y_pred_train = rf_model.predict(X_train_scaled)
    y_pred_test = rf_model.predict(X_test_scaled)
    y_prob = rf_model.predict_proba(X_test_scaled)[:,1]
    train_acc_score = accuracy_score(y_train_clf, y_pred_train)
    test_acc_score = accuracy_score(y_test_clf, y_pred_test)
    auc_score = roc_auc_score(y_test_clf, y_prob)
    print("\n ====== Random Forest Classifier ======")
    print(f"Train Accuracy: {train_acc_score:.4f}")
    print(f"Test Accuracy: {test_acc_score:.4f}")
    print(f"ROC-AUC score: {auc_score:.4f}")

    #Feature importance
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": rf_model.feature_importances_})
    importance_df = importance_df.sort_values(by="Importance",ascending=False)
    print("\n====== Top 5 important features ======")
    print(importance_df.head(5).to_string(index=False))
    return rf_model, train_acc_score, test_acc_score, auc_score, importance_df

def gradient_boosting_classifier(X_train_scaled,X_test_scaled,y_train_clf,y_test_clf):
    """
    Train a Gradient Boosting classifier and
    calculate training, test accuracy and ROC_AUC
    """
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train_scaled, y_train_clf)
    y_pred_train = gb_model.predict(X_train_scaled)
    y_pred_test = gb_model.predict(X_test_scaled)
    y_prob = gb_model.predict_proba(X_test_scaled)[:,1]
    train_acc_score = accuracy_score(y_train_clf, y_pred_train)
    test_acc_score = accuracy_score(y_test_clf, y_pred_test)
    auc_score = roc_auc_score(y_test_clf, y_prob)
    print("\n ====== Gradient Boosting Classifier ======")
    print(f"Train Accuracy: {train_acc_score:.4f}")
    print(f"Test Accuracy: {test_acc_score:.4f}")
    print(f"ROC-AUC score: {auc_score:.4f}")
    return gb_model, train_acc_score, test_acc_score, auc_score

def feature_ablation_study(X_train, X_test, y_train_clf, y_test_clf, importance_df):
    """
    Remove the 5 least important features and 
    compare Random Forest performance
    """
    least_features = importance_df.sort_values(by="Importance", ascending=True).head(5)["Feature"].tolist()
    print("\n====== Least Important Features ======")
    print(least_features)
    #Remove features
    X_train_reduced = X_train.drop(columns = least_features)
    X_test_reduced = X_test.drop(columns = least_features)
    #scale the reduced datset
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reduced)
    X_test_scaled = scaler.transform(X_test_reduced)

    rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_reduced.fit(X_train_scaled, y_train_clf)
    y_prob = rf_reduced.predict_proba(X_test_scaled)[:, 1]
    reduced_auc = roc_auc_score(y_test_clf, y_prob)
    return rf_reduced, reduced_auc, least_features



def main():
    os.makedirs("part3/output",exist_ok=True)
    df = load_data()
    if df is None:
        return
    X, y_reg, y_clf = feature_target_definition(df)
    X = pd.get_dummies(X, columns=["sender_domain"], drop_first=True)
    feature_names = X.columns.tolist()
    X_train,X_test,y_train_reg,y_test_reg,y_train_clf,y_test_clf = split_data(X, y_reg, y_clf)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    dt_model_base, base_train_acc_score, base_test_acc_score = decision_tree_baseline(X_train_scaled,X_test_scaled,y_train_clf,y_test_clf)
    dt_model_controlled, controlled_train_acc_score, controlled_test_acc_score = decision_tree_controlled(X_train_scaled,X_test_scaled,y_train_clf,y_test_clf)
    joblib.dump(dt_model_base, f"{OUTPUT_DIR}/decision_tree_baseline.pkl")
    joblib.dump(dt_model_controlled, f"{OUTPUT_DIR}/decision_tree_controlled.pkl")
    baseline_gap = base_train_acc_score - base_test_acc_score
    controlled_gap = controlled_train_acc_score - controlled_test_acc_score
    print("\n====== Train Test Differences ======")
    print(f"Baseline Tree: {baseline_gap:.4f}")
    print(f"Controlled Tree: {controlled_gap:.4f}\n")
    tree_comparison = pd.DataFrame({"Model": ["Baseline Tree","Controlled Tree"], "Train Accuracy": [base_train_acc_score, controlled_train_acc_score],
                               "Test Accuracy": [base_test_acc_score, controlled_test_acc_score]})
    tree_comparison.to_csv(f"{OUTPUT_DIR}/decision_tree_comparison.csv", index=False)
    print(tree_comparison.to_string(index=False))
    dt_model_gini, gini_test_acc_score = decision_tree_gini(X_train_scaled,X_test_scaled,y_train_clf,y_test_clf)
    dt_model_entropy, entropy_test_acc_score = decision_tree_entropy(X_train_scaled,X_test_scaled,y_train_clf,y_test_clf)
    joblib.dump(dt_model_gini, f"{OUTPUT_DIR}/decision_tree_gini.pkl")
    joblib.dump(dt_model_entropy, f"{OUTPUT_DIR}/decision_tree_entropy.pkl")
    print("\n")
    criterion_comparison = pd.DataFrame({"Criterion": ["Gini","Entropy"], "Test Accuracy": [gini_test_acc_score, entropy_test_acc_score]})
    criterion_comparison.to_csv(f"{OUTPUT_DIR}/criterion_comparison.csv", index=False)
    print(criterion_comparison.to_string(index=False))
    rf_model, rf_train_acc, rf_test_acc, rf_auc, importance_df = random_forest_classifier(X_train_scaled, X_test_scaled, y_train_clf, y_test_clf, feature_names)
    joblib.dump(rf_model, f"{OUTPUT_DIR}/random_forest.pkl")
    importance_df.to_csv(f"{OUTPUT_DIR}/random_forest_feature_importance.csv", index=False)
    gb_model, gb_train_acc, gb_test_acc, gb_auc = gradient_boosting_classifier(X_train_scaled,X_test_scaled,y_train_clf,y_test_clf)
    joblib.dump(gb_model, f"{OUTPUT_DIR}/gradient_boosting_classifier.pkl")
    print("\n")
    gb_comparison = pd.DataFrame({"Model": ["Gradient Boosting Classifier"], "Train Accuracy": [gb_train_acc], "Test Accuracy": [gb_test_acc], "ROC-AUC": [gb_auc]})
    gb_comparison.to_csv(f"{OUTPUT_DIR}/gradient_boosting_results.csv", index=False)
    print(gb_comparison.to_string(index=False))
    #unsclaed data is passed as the function scales reduced dataset after dropping the features
    rf_reduced, reduced_auc, least_features = feature_ablation_study(X_train, X_test, y_train_clf, y_test_clf, importance_df)
    ablation_comparison = pd.DataFrame({"Model": ["Full Random Forest", "Reduced Random Forest"], "ROC-AUC": [rf_auc, reduced_auc]})
    print("\n ====== Feature Ablation comparison ======")
    print(ablation_comparison.to_string(index=False))
    ablation_comparison.to_csv(f"{OUTPUT_DIR}/feature_ablation_comparison.csv")
    joblib.dump(rf_reduced, f"{OUTPUT_DIR}/random_forest_reduced.pkl")

if __name__ == "__main__":
    main()