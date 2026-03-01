# Customer Churn Prediction

A machine learning project that predicts whether a customer will churn from a subscription-based service. Built as part of my internship at **Uneeq Interns**.

---

## Overview

Customer churn is one of the most critical challenges for subscription businesses. This project explores multiple classification algorithms to identify customers likely to cancel their subscription, with a focus on metrics that matter from a business perspective — Precision, Recall, and F1 Score.

---

## Dataset

The dataset contains over 500,000 customer records with the following features:

| Feature | Description |
|---|---|
| Age | Customer age |
| Gender | Customer gender |
| Tenure | How long the customer has been subscribed |
| Usage Frequency | How often they use the service |
| Support Calls | Number of support calls made |
| Payment Delay | Days of payment delay |
| Subscription Type | Basic / Standard / Premium |
| Contract Length | Monthly / Quarterly / Annual |
| Total Spend | Total amount spent |
| Last Interaction | Days since last interaction |
| Churn | Target variable (1 = Churned, 0 = Not Churned) |

---

## Project Structure

```
├── T1.py                                          # Main script
├── requirements.txt                               # Dependencies
├── customer_churn_dataset-training-master.csv     # Training data
├── customer_churn_dataset-testing-master.csv      # Testing data
├── eda_plots.png                                  # EDA visualizations
├── model_comparison.png                           # Model comparison charts
└── best_model_analysis.png                        # Best model results
```

---

## Setup & Run

**1. Install dependencies**
```
pip install -r requirements.txt
```

**2. Run the script**
```
python T1.py
```

---

## What the Script Does

- Loads and explores the dataset (shape, types, missing values, churn distribution)
- Generates 6 EDA visualizations including churn distribution, age analysis, support calls impact, and a correlation heatmap
- Preprocesses data: drops nulls, encodes categorical features, applies standard scaling, and computes balanced class weights
- Trains 4 classification models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Evaluates each model on Precision, Recall, F1 Score, and AUC-ROC
- Saves ROC curves, Precision-Recall curves, confusion matrix, and feature importance plots

---

## Results

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| **Logistic Regression** | 0.5439 | 0.9766 | **0.6987** | **0.7748** |
| Decision Tree | 0.4943 | 0.9981 | 0.6612 | 0.5395 |
| Random Forest | 0.4909 | 0.9982 | 0.6582 | 0.7247 |
| Gradient Boosting | 0.4881 | 0.9987 | 0.6557 | 0.6917 |

**Best Model: Logistic Regression**

With a Recall of 0.977, the model correctly identifies almost all customers who will churn. In a business context, missing a churner is far more costly than a false alarm, which is why high recall was prioritized.

The most influential features were **Support Calls**, **Total Spend**, and **Payment Delay**.

---

## Key Insights

- Customers with more support calls are significantly more likely to churn
- Monthly contract customers churn at a much higher rate than annual ones
- Total spend and payment delay are strong predictors of churn behavior
- Subscription type had minimal impact on churn rate across all tiers

---

## Tech Stack

- Python 3.12
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
