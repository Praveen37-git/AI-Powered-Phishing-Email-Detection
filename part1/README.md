# Part 1 – Data Acquisition, Cleaning, and Exploratory Data Analysis

## Project Overview

This project focuses on the preprocessing and exploratory analysis of a phishing website detection dataset. The objective is to understand the dataset, perform data cleaning, analyze feature distributions, identify potential outliers, and generate visualizations that will support machine learning tasks in later parts of the capstone.

---

## Dataset

Dataset Name: Phishing Website Detector

Source:
https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector

The dataset contains **11,054 records** and **32 columns**.

Each row represents a website with features describing its characteristics. The target column (`class`) indicates whether the website is phishing (`-1`) or legitimate (`1`).

---

## Data Loading

The dataset was loaded using `pandas.read_csv()`.

The following information was displayed:

- First five rows
- Dataset shape
- Column names
- Data types

---

## Missing Value Analysis

The number and percentage of missing values were calculated for every column.

### Findings

- No missing values were present in the dataset.
- Therefore, no columns exceeded the 20% missing value threshold.
- Median imputation was implemented in the code to satisfy the assignment requirements but was not applied because no missing values existed.

### Why Median?

Median is preferred over the mean for skewed distributions because it is less affected by extreme values and provides a better measure of central tendency.

---

## Duplicate Analysis

Duplicate records were identified using:

```python
df.duplicated().sum()
```

### Findings

- No duplicate records were found.
- Therefore, no rows were removed.

---

## Data Type Optimization

All columns were automatically detected as numeric (`int64`) by pandas.

Since the dataset contains encoded categorical values, the target column (`class`) was converted to the `category` data type to reduce memory usage.

### Memory Usage

Memory usage was measured before and after optimization, showing a reduction after converting the target column.

---

## Descriptive Statistics

Descriptive statistics including:

- Count
- Mean
- Standard deviation
- Minimum
- Maximum
- Quartiles

were generated using `describe()`.

---

## Skewness Analysis

Skewness was calculated for every numeric feature.

The feature with the highest absolute skewness was identified.

The two most skewed features were also analyzed by comparing their mean and median values.

### Observation

The median provides a better representation of central tendency for highly skewed variables because the mean is influenced by extreme observations.

---

## Outlier Detection

The Interquartile Range (IQR) method was applied to:

- LongURL
- PageRank

For each feature, the following values were calculated:

- Q1
- Q3
- IQR
- Lower Bound
- Upper Bound
- Number of Outliers

### Observation

No outliers were removed because these values represent valid encoded phishing website characteristics rather than data entry errors.

Removing them could eliminate useful information for machine learning.

---

# Exploratory Data Analysis

## Line Plot

A rolling average of Website Traffic was plotted over the first 200 observations.

### Observation

The rolling average smooths the discrete encoded values and provides a clearer visualization of changes across the dataset.

---

## Bar Chart

The average Website Traffic was compared across website classes.

### Observation

The phishing and legitimate classes show different average Website Traffic values, suggesting this feature may provide predictive information.

---

## Histogram

A histogram was generated for the most skewed feature.

### Observation

The distribution is highly skewed because the feature values are encoded into a small number of discrete categories (-1, 0, and 1).

---

## Scatter Plot

A scatter plot was generated between:

- WebsiteTraffic
- PageRank

### Observation

The relationship appears weak because both variables contain discrete encoded values rather than continuous measurements.

---

## Box Plot

A box plot was created for WebsiteTraffic grouped by website class.

### Observation

The distributions differ between phishing and legitimate websites, indicating WebsiteTraffic may be useful during classification.

---

## Correlation Analysis

A Pearson correlation matrix was generated and visualized using a heatmap.

The pair of features with the strongest absolute correlation was identified.

### Observation

High correlation indicates a strong statistical relationship but does not imply causation.

The correlation may exist because both variables describe similar characteristics of phishing websites.

---

## Pearson vs Spearman Correlation

Pearson and Spearman correlation matrices were computed and compared.

The absolute differences between the two matrices were calculated to identify feature pairs where linear and monotonic relationships differ.

### Observation

Pairs with larger differences indicate monotonic but non-linear relationships.

These relationships will be considered during feature selection in later stages of the project.

---

## Grouped Aggregation

The dataset was grouped using the target class.

For each group, the following statistics were calculated for WebsiteTraffic:

- Mean
- Standard Deviation
- Count

### Observation

The grouped statistics demonstrate differences between phishing and legitimate websites and provide insight into the predictive capability of WebsiteTraffic.

---

## Output

The cleaned dataset was exported as:

```
cleaned_data.csv
```

This dataset will be used in Parts 2 and 3 of the capstone project.

---

## Libraries Used

- pandas
- matplotlib
- seaborn
- os

---

## Conclusion

The dataset required minimal preprocessing because it contained no missing values or duplicate records. Exploratory analysis showed meaningful differences between phishing and legitimate websites across several engineered features. The cleaned dataset and generated visualizations provide a solid foundation for developing supervised machine learning models in the next phase of the project.