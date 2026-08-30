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
- [x] Scale Amount and Time Features
- [x] Train-Test Split with Stratification
- [x] Baseline Logistic Regression
- [x] Handle Class Imbalance using Class Weights
- [x] Handle Class Imbalance using SMOTE
- [x] Random Forest Classifier
- [x] Model Evaluation and Comparison
- [x] Confusion Matrix
- [x] ROC Curve and ROC-AUC
- [x] Precision-Recall Curve and PR-AUC
- [x] Decision Threshold Tuning
- [x] Final Model Evaluation

### Remaining

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

A total of 1,081 duplicate rows were identified during the initial dataset inspection. These duplicates were considered during the preprocessing stage rather than being removed automatically.

### Class Imbalance

The dataset is extremely imbalanced:

| Class | Transaction Type | Count | Percentage |
|------:|------------------|------:|-----------:|
| 0 | Legitimate | 284,315 | 99.8273% |
| 1 | Fraudulent | 492 | 0.1727% |

There are roughly 578 legitimate transactions for every fraudulent transaction.

This severe imbalance is one of the main challenges of the project.

### Transaction Amount

The `Amount` feature contains transaction values with a wide range and a skewed distribution.

The transaction amount distributions of legitimate and fraudulent transactions show differences, but there is still overlap between the two classes. Therefore, transaction amount alone cannot reliably identify fraudulent transactions.

### Correlation Analysis

Correlation analysis was performed to understand the linear relationship between the numerical features and the target variable.

The anonymized features (`V1` to `V28`) show different levels of correlation with the `Class` variable. Some features have stronger relationships with the target, while others have relatively weak linear relationships.

Correlation was treated as an exploratory tool rather than as the sole basis for feature selection.

---

## Key Findings from EDA

The EDA highlighted several important characteristics of the dataset:

1. Fraudulent transactions are extremely rare.
2. The dataset is highly imbalanced.
3. Accuracy alone would not be a reliable measure of fraud detection performance.
4. Transaction amounts have a wide and skewed distribution.
5. The anonymized features have different relationships with the fraud target.
6. Appropriate preprocessing and imbalance-handling techniques are important for building a useful fraud detection model.

---

## Handling Class Imbalance

Three approaches were explored to understand how different imbalance-handling strategies affect model performance:

### 1. Baseline Logistic Regression

The first Logistic Regression model was trained without any special imbalance-handling technique.

This provided a baseline against which the other approaches could be compared.

### 2. Class-Weighted Logistic Regression

Class weights were used to give more importance to fraudulent transactions during model training.

This significantly increased fraud recall, but also produced a large number of false positives.

### 3. SMOTE Logistic Regression

SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data to generate synthetic examples of the minority fraud class.

The test set was kept unchanged to ensure that model evaluation remained representative of the original data distribution.

---

## Model Evaluation and Comparison

Four models were evaluated:

- Baseline Logistic Regression
- Class-Weighted Logistic Regression
- SMOTE Logistic Regression
- Random Forest

The following results were obtained on the test set:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Logistic Regression | 0.9992 | 0.8289 | 0.6429 | 0.7241 | 0.9559 | 0.7432 |
| Class-Weighted Logistic Regression | 0.9755 | 0.0610 | 0.9184 | 0.1144 | 0.9722 | 0.7159 |
| SMOTE Logistic Regression | 0.9742 | 0.0580 | 0.9184 | 0.1092 | 0.9699 | 0.7249 |
| Random Forest | 0.9996 | 0.9412 | 0.8163 | 0.8743 | 0.9630 | 0.8734 |

These results show that the different imbalance-handling approaches produce significantly different precision-recall trade-offs.

The class-weighted and SMOTE Logistic Regression models achieved higher recall but produced many false positives.

Random Forest provided a much stronger balance between precision and recall at the default threshold.

---

## Confusion Matrix Analysis

The confusion matrices provided additional insight into the types of errors made by each model.

### Baseline Logistic Regression

```text
[[56851    13]
 [   35    63]]

