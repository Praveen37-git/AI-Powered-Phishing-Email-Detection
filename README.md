# AI-Powered Phishing Email Detection

This repository contains my submission for the **Applied AI & ML Essentials Capstone Project**. The project demonstrates an end-to-end machine learning workflow for phishing email detection, starting from data preprocessing and exploratory data analysis to advanced machine learning models and LLM-powered explanations.

## Dataset

- **Dataset:** CEAS 2008 Spam Email Dataset
- **Source:** https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset
- **Records:** 39,154 emails
- **Task:** Binary classification (Phishing/Spam vs Legitimate)

## Project Structure

```
part1/
├── Data Cleaning & Exploratory Data Analysis

part2/
├── Supervised Machine Learning

part3/
├── Advanced Machine Learning & Ensemble Models

part4/
├── LLM-Powered Phishing Email Detection
```

## What I Built

### Part 1 – Data Preparation
- Data cleaning and preprocessing
- Feature engineering
- Exploratory Data Analysis (EDA)
- Data visualization

### Part 2 – Machine Learning
- Linear Regression
- Ridge Regression
- Logistic Regression
- ROC Curve analysis
- Model evaluation

### Part 3 – Advanced Modeling
- Decision Tree
- Random Forest
- Gradient Boosting
- Cross-validation
- Feature importance
- GridSearchCV
- Learning curves
- Model serialization

### Part 4 – LLM Integration
- Loaded the trained Random Forest model
- Generated phishing predictions
- Used an LLM to explain model predictions
- JSON schema validation
- PII guardrail using regex
- Tested multiple email scenarios and temperature settings
- Saved structured JSON outputs

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- OpenRouter API (GPT-4.1 Mini)
- JSON Schema
- Requests

## Repository Setup

```bash
git clone https://github.com/Praveen37-git/AI-Powered-Phishing-Email-Detection.git
cd AI-Powered-Phishing-Email-Detection
pip install -r requirements.txt
```

## Project Status

- ✅ Part 1 – Completed
- ✅ Part 2 – Completed
- ✅ Part 3 – Completed
- ✅ Part 4 – Completed

## Author

**Praveen Kumar S**