import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Assign the dataset and create a output file to store cleaned and processed dataset
FILE_PATH = "part1/data/phishing.csv"
OUTPUT_DIR = "part1/output"

#load the data from the file
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

#Calculate missing values and percentage
def missing_value_analysis(df):
    missing_count = df.isnull().sum()
    missing_percentage = ((missing_count / len(df)) * 100).round(2)
    missing_table = pd.DataFrame({"Missing count": missing_count, "Missing percentage": missing_percentage})
    print(missing_table)
    return missing_percentage

#remove duplicate values from the datset
def remove_duplicates(df):
    duplicates_count = df.duplicated().sum()
    print(f"\nNumber of Duplicate rows in the dataset: {duplicates_count}")
    cleaned_df = df.drop_duplicates()
    print(f"\nShape after removing duplicates: {cleaned_df.shape}")
    print(f"Duplicate rows removed: {duplicates_count}")
    return cleaned_df

#handling missing/NaN values
def handle_missing_values(df):
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        if pd.api.types.is_numeric_dtype(df[column]) and 0 < missing_percentage < 20:
            df[column] = df[column].fillna(df[column].median())
            print(f"Filled missing values in {column} using median.")
        elif missing_percentage > 20:
            print(f"\nWarning!!! {column} has {missing_percentage:.2f}% missing values")
    print("\nRemaining missing values:")
    print(df.isnull().sum())
    return df

#data types modification, assigning categorical columns
def correct_data_types(df):
    memory_before = df.memory_usage(deep = True).sum()
    print(f"Memory before optimization: {memory_before} bytes")
    categorical_columns = ["class"]
    for column in categorical_columns:
        df[column] = df[column].astype("category")
    memory_after = df.memory_usage(deep = True).sum()
    print(f"Memory after optimization: {memory_after} bytes")
    memory_saved = memory_before - memory_after
    print(f"Memory saved: {memory_saved} bytes")
    return df

#calculate skewness, most_skewed_column and mean, median comparison
def descriptive_statistics(df):
    numeric_df = df.select_dtypes(include=['number'])
    print("\n===== Descriptive Statistics =====")
    print(numeric_df.describe())
    skewness = {}
    for column in numeric_df.columns:
        skewness[column] = numeric_df[column].skew()
    print("\n===== Skewness =====")
    for column, value in skewness.items():
        print(f"Skewness for {column} is {value:.2f}")
    most_skewed_column = max(skewness,key=lambda column: abs(skewness[column])
    )
    print(f"\nMost skewed column: {most_skewed_column}")
    print(f"Skewness value: {skewness[most_skewed_column]:.2f}")
    sorted_skew = sorted(skewness.items(),key=lambda item: abs(item[1]),reverse=True)
    top_two = sorted_skew[:2]
    print("\n===== Mean vs Median Comparison =====")
    for column, skew in top_two:
        print(f"\nColumn: {column}")
        print(f"Skewness: {skew:.2f}")
        print(f"Mean: {df[column].mean():.2f}")
        print(f"Median: {df[column].median():.2f}")
    return {
        "skewness": skewness,
        "most_skewed_column": most_skewed_column,
        "most_skewed_value": skewness[most_skewed_column]
    }

#detect outliers in the dataset
def detect_outliers(df, column):
    print("\n=====Outlier Analysis=====")
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outlier_count = len(outliers)
    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Outliers found: {outlier_count}")
    return {
    "Q1": Q1,
    "Q3": Q3,
    "IQR": IQR,
    "Lower Bound": lower_bound,
    "Upper Bound": upper_bound,
    "Outlier Count": outlier_count
    }

#using different functions for each graph
def create_visualizations(df, most_skewed_column):
    plot_line(df)
    plot_bar(df)
    plot_histogram(df, most_skewed_column)
    plot_scatter(df)
    plot_box(df)    
    return df

#Line chart with Index and Website Traffic Average
def plot_line(df):
    line_chart_df = df.head(200)
    rolling_mean = line_chart_df["WebsiteTraffic"].rolling(window = 10, min_periods = 1).mean()
    plt.figure(figsize=(10,5))
    plt.plot(line_chart_df["Index"], rolling_mean)
    plt.title("Rolling Average of Website Traffic")
    plt.xlabel("Index")
    plt.ylabel("Rolling Average")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/line_plot.png")
    plt.show()

#bar chart with Website Traffic and class columns
def plot_bar(df):
    grouped_data = (df.groupby("class")["WebsiteTraffic"].mean())
    plt.figure(figsize=(6,5))
    grouped_data.plot(kind = "bar", rot = 0)
    plt.title("Average website traffic by class")
    plt.xlabel("Class")
    plt.ylabel("Average Website Traffic")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/bar_chart.png")
    plt.show()

#plotting histogram with the distribution of the given column
def plot_histogram(df, column):
    plt.figure(figsize=(8,5))
    sns.histplot(df[column], bins=20)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/histogram.png")
    plt.show()

#scatter plot with Website Traffic and PageRank
def plot_scatter(df):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df,x="WebsiteTraffic",y="PageRank")
    plt.title("Website Traffic vs PageRank")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scatter_plot.png")    
    plt.show()

#box chart with class and Website Traffic
def plot_box(df):
    plt.figure(figsize=(7,5))
    sns.boxplot(data=df,x="class",y="WebsiteTraffic")
    plt.title("Website Traffic by class")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/box_plot.png")
    plt.show()

#correlation analsyis - pearson and spearman and creating heatmap
def correlation_analysis(df):
    numeric_df = df.select_dtypes(include = ["number"])
    pearson = numeric_df.corr()
    spearman = numeric_df.corr(method = "spearman")
    plt.figure(figsize=(16,12))
    sns.heatmap(pearson, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Pearson Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
    plt.show()
    correlation = pearson.abs().copy()
    # Remove self-correlations
    for column in correlation.columns:
        correlation.loc[column, column] = 0
    highest_pair = correlation.stack().idxmax()
    highest_value = correlation.stack().max()
    print(f"\nHighest correlation:")
    print(f"{highest_pair[0]} <-> {highest_pair[1]}")
    print(f"Correlation = {highest_value:.2f}")
    return {
        "pearson": pearson,
        "spearman": spearman
    }


def grouped_aggregation(df):
    print("\n===== Grouped Aggregation =====")
    grouped = (df.groupby("class")["WebsiteTraffic"].agg(["mean","std","count"]))
    lowest_mean = grouped["mean"].min()
    if lowest_mean <= 0:
        print("Ratio can't be computed because minimum group mean is zero or negative.")
    else:
        ratio = grouped["mean"].max() / lowest_mean
        print(f"Mean ratio: {ratio:.2f}")
    print(grouped)
    return grouped

#find the top correlation differences between pearson and spearman methods
def compare_correlations(pearson, spearman):
    difference = (spearman - pearson).abs()
    difference = difference.where(~difference.index.to_series().apply(lambda x: difference.columns == x), 0)
    print("\nTop correlation differences: ")
    print(difference.stack().sort_values(ascending = False).head(3))
    return difference

#save the modified, cleaned datset in separate file
def save_cleaned_data(df):
    df.to_csv("part1/output/cleaned_data.csv", index = False)
    print("cleaned_data.csv created successfully.")
    return df

def main():
    #making sure the output directory is existing
    os.makedirs(OUTPUT_DIR, exist_ok= True)
    df = load_data()
    if df is None:
        return
    explore_data(df)
    missing_value_analysis(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = correct_data_types(df)
    stats = descriptive_statistics(df)
    detect_outliers(df, "LongURL")
    detect_outliers(df, "PageRank")
    create_visualizations(df, stats["most_skewed_column"])
    correlation_results = correlation_analysis(df)
    compare_correlations(
        correlation_results["pearson"],
        correlation_results["spearman"]
    )
    grouped_aggregation(df)
    save_cleaned_data(df)


if __name__ == "__main__":
    main()




