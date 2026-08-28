# FraudGuard — Credit Card Fraud Detection

## Project Overview

FraudGuard is a machine learning project focused on detecting fraudulent credit card transactions from highly imbalanced data.

Credit card fraud detection is a challenging classification problem because fraudulent transactions are extremely rare compared with legitimate transactions. A model can achieve very high accuracy by simply predicting most transactions as legitimate, while still failing to detect actual fraud.

The goal of this project is to build and evaluate machine learning models that can identify fraudulent transactions while focusing on meaningful metrics such as precision, recall, F1-score, ROC-AUC, and Precision-Recall AUC.

---

## Dataset

The project uses the publicly available **Credit Card Fraud Detection** dataset from Kaggle.

Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains:

- 284,807 transactions
- 31 columns
- 492 fraudulent transactions
- 284,315 legitimate transactions
- Approximately 0.17% of transactions are fraudulent

The target variable is:

- `Class = 0` → Legitimate transaction
- `Class = 1` → Fraudulent transaction

Most of the transaction features are anonymized as `V1` to `V28`.

---

## Project Objectives

The main objectives of this project are:

- Understand the structure of the transaction data
- Perform exploratory data analysis
- Analyze the severe class imbalance
- Understand why accuracy is misleading for fraud detection
- Scale the required numerical features
- Split the data using stratification
- Build a baseline Logistic Regression model
- Handle class imbalance using class weights
- Experiment with SMOTE oversampling
- Train a Random Forest classifier
- Compare model performance
- Tune the classification decision threshold
- Select and evaluate the final model
- Interpret the model and analyze prediction errors

---

## Tools and Technologies

- Python 3.x
- Jupyter Notebook / VS Code
- pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- imbalanced-learn

---

## Project Progress

### Completed

- [x] Project Overview
- [x] Import Required Libraries
- [x] Load the Dataset
- [x] Understand the Dataset
- [x] Exploratory Data Analysis (EDA)
- [x] Analyze Class Imbalance
- [x] Understand Why Accuracy Is Misleading

### In Progress

- [ ] Train-Test Split with Stratification
- [ ] Scale Amount and Time Features
- [ ] Baseline Logistic Regression
- [ ] Handle Class Imbalance using Class Weights
- [ ] Handle Class Imbalance using SMOTE
- [ ] Random Forest Classifier
- [ ] Model Evaluation and Comparison
- [ ] Confusion Matrix
- [ ] ROC Curve and ROC-AUC
- [ ] Precision-Recall Curve and PR-AUC
- [ ] Decision Threshold Tuning
- [ ] Final Model Evaluation
- [ ] Final Model Selection and Justification
- [ ] Feature and Model Interpretation
- [ ] Error Analysis
- [ ] Model Limitations and Possible Improvements
- [ ] Conclusion and Key Findings

---

## Exploratory Data Analysis

The initial analysis was performed to understand the structure and characteristics of the dataset before building machine learning models.

### Dataset Structure

The dataset contains 284,807 transactions and 31 columns.

All columns contain values for every transaction, so no missing-value treatment is required at this stage.

A total of 1,081 duplicate rows were identified during the initial dataset inspection. These duplicates will be considered carefully during the preprocessing stage rather than being removed automatically.

### Class Imbalance

The dataset is extremely imbalanced:

| Class | Transaction Type | Count | Percentage |
|------:|------------------|------:|-----------:|
| 0 | Legitimate | 284,315 | 99.8273% |
| 1 | Fraudulent | 492 | 0.1727% |

There are roughly 578 legitimate transactions for every fraudulent transaction.

This severe imbalance is one of the main challenges of the project.

### Transaction Amount

The `Amount` feature contains transaction values with a wide range. The distribution also contains large values, making it useful to examine the feature using a logarithmic scale.

When comparing transaction amounts between legitimate and fraudulent transactions, the distributions show differences, but there is still overlap between the two classes. This means transaction amount alone cannot reliably identify fraud.

### Correlation Analysis

Correlation analysis was performed to understand the linear relationship between the numerical features and the target variable.

The anonymized features (`V1` to `V28`) show different levels of correlation with the `Class` variable. Some features have stronger relationships with the target, while others have relatively weak linear relationships.

Correlation will be treated as an exploratory tool rather than as the sole basis for feature selection.

---

## Key Findings So Far

The EDA highlighted several important characteristics of the dataset:

1. Fraudulent transactions are extremely rare.
2. The dataset is highly imbalanced.
3. Accuracy alone would not be a reliable measure of fraud detection performance.
4. Transaction amounts have a wide and skewed distribution.
5. The anonymized features have different relationships with the fraud target.
6. Appropriate preprocessing and imbalance-handling techniques will be important for building useful models.

---

## Evaluation Strategy

Because this is a highly imbalanced classification problem, model performance will be evaluated primarily using:

- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall AUC
- Confusion Matrix

Accuracy will not be treated as the primary performance metric because it can be misleading when the minority class is extremely small.

---

## Project Structure

```text
FraudGuard/
│
├── FraudGuard.ipynb
├── creditcard.csv
└── README.md