# FraudGuard — Credit Card Fraud Detection

## Project Overview

FraudGuard is a machine learning project focused on detecting fraudulent credit card transactions from a highly imbalanced dataset.

The main challenge in this project is that fraud is very rare. Out of 284,807 transactions, only 492 are fraudulent. Because of this, a model can get a very high accuracy score simply by predicting most transactions as legitimate while still missing fraud cases.

Instead of focusing only on accuracy, this project uses metrics such as precision, recall, F1-score, ROC-AUC, Precision-Recall AUC, and confusion matrices.

Different approaches for handling class imbalance were also tested, including class weighting and SMOTE. Logistic Regression and Random Forest were then compared, followed by decision threshold tuning for the final model.

---

## Dataset

The project uses the publicly available **Credit Card Fraud Detection** dataset from Kaggle.

Dataset:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dataset Details

| Detail | Value |
|---|---:|
| Total Transactions | 284,807 |
| Legitimate Transactions | 284,315 |
| Fraudulent Transactions | 492 |
| Legitimate Transactions | 99.8273% |
| Fraudulent Transactions | 0.1727% |
| Total Columns | 31 |

The target column is `Class`:

- `0` → Legitimate transaction
- `1` → Fraudulent transaction

Most of the transaction features are anonymized and represented as `V1` to `V28`.

---

## Project Objectives

Through this project, I aimed to:

- Understand the structure of the dataset
- Perform exploratory data analysis
- Analyze the severe class imbalance
- Understand why accuracy can be misleading
- Scale the `Time` and `Amount` features
- Perform a stratified train-test split
- Build a baseline Logistic Regression model
- Try class weighting to handle imbalance
- Try SMOTE oversampling
- Train a Random Forest model
- Compare the different approaches
- Study confusion matrices
- Compare ROC-AUC and PR-AUC
- Tune the decision threshold
- Evaluate the final model
- Understand the model's errors and limitations

---

## Tools and Libraries

The project was developed using:

- Python 3.x
- VS Code
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- imbalanced-learn

---


## Notebook Structure

- Project Overview
- Import Required Libraries
- Load the Dataset
- Understand the Dataset
- Exploratory Data Analysis (EDA)
- Analyze Class Imbalance
- Understand Why Accuracy Is Misleading
- Scale Amount and Time Features
- Train-Test Split with Stratification
- Baseline Logistic Regression
- Handle Class Imbalance using Class Weights
- Handle Class Imbalance using SMOTE
- Random Forest Classifier
- Model Evaluation and Comparison
- Confusion Matrix
- ROC Curve and ROC-AUC
- Precision-Recall Curve and PR-AUC
- Decision Threshold Tuning
- Final Model Evaluation
- Final Model Selection and Justification
- Feature and Model Interpretation
- Error Analysis
- Model Limitations and Possible Improvements
- Conclusion and Key Findings

---

## Project Structure

The project is organized into separate folders for easier access to the dataset, notebook, visualizations, outputs, and summary.

```text
FraudGuard/
│
├── Data/
│   └── creditcard.csv
│
├── Images/
│   └── Project visualizations
│
├── Notebooks/
│   └── Credit_card_fraud_detection.ipynb
│
├── Outputs/
│   └── Important analysis and model results
│
├── Summary/
│   └── credit_card_fraud_detection_summary.pdf
│
├── .gitignore
└── README.md