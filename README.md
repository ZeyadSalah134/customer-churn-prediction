# Customer Churn Prediction

Predicting customer churn for a subscription-based service using classification algorithms.

## Dataset

Download the dataset from Kaggle and place both CSV files in the same folder as the script:

- `customer_churn_dataset-training-master.csv`
- `customer_churn_dataset-testing-master.csv`

## Setup

Make sure Python is installed, then install dependencies:

```
pip install -r requirements.txt
```

## Run

```
python customer_churn_prediction.py
```

## What it does

- Exploratory data analysis with visualizations
- Preprocessing: label encoding, standard scaling, class weight balancing
- Trains and evaluates 4 models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Evaluates using Precision, Recall, F1 Score and AUC-ROC
- Saves 3 output plots: `eda_plots.png`, `model_comparison.png`, `best_model_analysis.png`

## Results

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Logistic Regression | 0.5439 | 0.9766 | 0.6987 | 0.7748 |
| Decision Tree | 0.4943 | 0.9981 | 0.6612 | 0.5395 |
| Random Forest | 0.4909 | 0.9982 | 0.6582 | 0.7247 |
| Gradient Boosting | 0.4881 | 0.9987 | 0.6557 | 0.6917 |

Best model: **Logistic Regression** (F1 = 0.6987, AUC = 0.7748)
