import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Get the cleaned dataset  from part1
FILE_PATH = "part2/data/cleaned_data.csv"

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
    dropping the target columns from the dataset
    dropping index column as it's just a row-identifier and
    it doesn't have any information for prediction
    """
    
    X = df.drop(columns = ["class","DomainRegLen","Index"])
    #Regression target column
    y_reg = df["DomainRegLen"]
    #classification target column
    y_clf = df["class"]
    return X,y_reg,y_clf

def split_data(X,y_reg,y_clf):
    """
    Splitting the dataset into train and test sets for 
    both regression and classification models
    """
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(X,y_reg,y_clf,test_size=0.2,random_state=42)

    return X_train,X_test,y_train_reg,y_test_reg,y_train_clf,y_test_clf

def scale_data(X_train,X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def main():
    df = load_data()
    if df is None:
        return
    explore_data(df)
    X,y_reg,y_clf = feature_target_definition(df)
    print(f"X: {X.shape}")
    print(f"y_reg: {y_reg.shape}")
    print(f"y_clf: {y_clf.shape}")
    X_train,X_test,y_train_reg,y_test_reg,y_train_clf,y_test_clf = split_data(X,y_reg,y_clf)
    X_train_scaled, X_test_scaled = scale_data(X_train,X_test)


if __name__ == "__main__":
    main()