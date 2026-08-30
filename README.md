# FraudGuard — Credit Card Fraud Detection

## Project Overview

FraudGuard is a machine learning project where I worked on detecting fraudulent credit card transactions from a highly imbalanced dataset.

The main challenge in this project is that fraud is very rare. Out of 284,807 transactions, only 492 are fraudulent. Because of this, a model can get a very high accuracy score simply by predicting almost every transaction as legitimate, while still missing many fraud cases.

So, instead of focusing only on accuracy, I explored metrics such as precision, recall, F1-score, ROC-AUC, Precision-Recall AUC, and the confusion matrix.

I also experimented with different ways of handling class imbalance and compared Logistic Regression with Random Forest before tuning the final decision threshold.

---

## Dataset

I used the publicly available **Credit Card Fraud Detection** dataset from Kaggle.

Dataset:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains:

- 284,807 transactions
- 31 columns
- 492 fraudulent transactions
- 284,315 legitimate transactions
- Only about 0.17% of the transactions are fraudulent

The target column is `Class`:

- `0` → Legitimate transaction
- `1` → Fraudulent transaction

Most of the transaction features are anonymized and are represented as `V1` to `V28`.

---

## Project Objectives

Through this project, I aimed to:

- Understand the structure of the dataset
- Perform exploratory data analysis
- Understand the severe class imbalance
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
- VS Code / Jupyter Notebook
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

# Exploratory Data Analysis

Before training any models, I first explored the dataset to understand what kind of data I was working with.

### Dataset Structure

The dataset contains 284,807 transactions and 31 columns.

All columns had complete data, so there were no missing values that needed to be handled.

During the initial inspection, I found 1,081 duplicate rows. I did not remove them automatically because duplicate transactions can be meaningful in a fraud detection dataset and should be considered carefully rather than removed without understanding their context.

### Class Imbalance

The biggest issue I found during EDA was the huge difference between legitimate and fraudulent transactions.

| Class | Transaction Type | Count | Percentage |
|------:|------------------|------:|-----------:|
| 0 | Legitimate | 284,315 | 99.8273% |
| 1 | Fraudulent | 492 | 0.1727% |

This means there are roughly 578 legitimate transactions for every fraudulent transaction.

This imbalance became one of the main challenges I had to deal with during the modelling stage.

### Transaction Amount

The `Amount` feature has a wide and skewed distribution.

When comparing the transaction amounts of legitimate and fraudulent transactions, I could see some differences between the two groups. However, there was still considerable overlap.

This showed that transaction amount by itself would not be enough to reliably identify fraud.

### Correlation Analysis

I also looked at the correlations between the numerical features and the `Class` column.

The anonymized features from `V1` to `V28` showed different levels of correlation with the target. Some features had stronger relationships with fraud, while others had weaker relationships.

I treated correlation as an exploratory tool rather than using it alone to decide which features should be included in the model.

---

# Key Findings from EDA

The EDA helped me understand a few important things before moving into modelling:

1. Fraudulent transactions are extremely rare.
2. The dataset is heavily imbalanced.
3. Accuracy by itself would not give a useful picture of model performance.
4. Transaction amounts have a wide and skewed distribution.
5. The anonymized features have different relationships with the target.
6. Handling the class imbalance would be an important part of the modelling process.

---

# Handling Class Imbalance

Since fraud represented only 0.1727% of the dataset, I experimented with different approaches to deal with the imbalance.

### Baseline Logistic Regression

I first trained a Logistic Regression model without applying any special imbalance-handling technique.

This gave me a baseline that I could use to understand how the model behaved before making any changes.

The baseline model achieved:

- Precision: 82.89%
- Recall: 64.29%
- F1-score: 72.41%
- ROC-AUC: 95.59%
- PR-AUC: 74.32%

Although the accuracy was 99.92%, the recall showed that the model was still missing a significant number of fraudulent transactions.

### Class-Weighted Logistic Regression

Next, I used class weights so that fraudulent transactions would receive more importance during training.

This increased fraud recall from 64.29% to 91.84%.

However, precision dropped significantly to 6.10%.

In other words, the model became much more aggressive about detecting fraud, but it also started incorrectly flagging many legitimate transactions.

### SMOTE Logistic Regression

I then tried SMOTE, which stands for Synthetic Minority Over-sampling Technique.

SMOTE creates synthetic examples of the minority class instead of simply duplicating existing fraud transactions.

I applied SMOTE only to the training data and kept the test set unchanged.

The SMOTE model achieved:

- Precision: 5.80%
- Recall: 91.84%
- F1-score: 10.92%
- ROC-AUC: 96.99%
- PR-AUC: 72.49%

The model detected more fraud cases, but again produced a very large number of false positives.

---

# Model Comparison

After experimenting with Logistic Regression and different imbalance-handling methods, I trained a Random Forest model and compared all four approaches.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Logistic Regression | 0.9992 | 0.8289 | 0.6429 | 0.7241 | 0.9559 | 0.7432 |
| Class-Weighted Logistic Regression | 0.9755 | 0.0610 | 0.9184 | 0.1144 | 0.9722 | 0.7159 |
| SMOTE Logistic Regression | 0.9742 | 0.0580 | 0.9184 | 0.1092 | 0.9699 | 0.7249 |
| Random Forest | 0.9996 | 0.9412 | 0.8163 | 0.8743 | 0.9630 | 0.8734 |

The results were interesting because there was no single metric that made every model look best.

The class-weighted Logistic Regression model had the highest ROC-AUC, but its precision was extremely low.

Random Forest, on the other hand, had the highest PR-AUC and provided a much better balance between precision and recall.

---

# Confusion Matrix Analysis

I used confusion matrices to understand exactly what kinds of mistakes each model was making.

### Baseline Logistic Regression

```text
[[56851    13]
 [   35    63]]