# Part 1 – Data Cleaning and Exploratory Data Analysis

## Dataset

This project uses the **CEAS 2008 Spam Email Dataset** downloaded from Kaggle.

Download the dataset from:
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

Place the dataset in:

```
part1/data/CEAS_08.csv
```

---

## What was done

- Loaded and explored the dataset.
- Checked for missing values and duplicate records.
- Filled missing values in `receiver` and `subject` using the mode.
- Converted `sender`, `receiver`, and `label` to categorical data types to reduce memory usage.
- Created new features from the email text:
  - subject_length
  - body_length
  - word_count
  - uppercase_count
  - special_characters
  - question_marks
  - exclamation_marks
  - contains_http
- Removed the duplicate `url_count` feature before correlation analysis.
- Detected outliers using the IQR method (kept all outliers since they represent valid email characteristics).
- Generated descriptive statistics and skewness analysis.
- Compared Pearson and Spearman correlations.
- Performed grouped aggregation based on email labels.
- Saved the cleaned dataset as `cleaned_data.csv`.

---

## Visualizations

The following plots are generated:

- Line Plot
- Bar Chart
- Histogram
- Scatter Plot
- Box Plot
- Correlation Heatmap

All plots and the cleaned dataset are saved in the `output` folder.

---

## Libraries Used

- pandas
- numpy
- matplotlib
- seaborn

---

## Run

```bash
python main.py
```