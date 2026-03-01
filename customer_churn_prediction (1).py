import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, precision_score, recall_score,
                             average_precision_score)
from sklearn.utils.class_weight import compute_class_weight


train_df = pd.read_csv('customer_churn_dataset-training-master.csv')
test_df  = pd.read_csv('customer_churn_dataset-testing-master.csv')

print(f"Training set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"Testing set:  {test_df.shape[0]} rows, {test_df.shape[1]} columns")
print(f"\nColumn types:\n{train_df.dtypes}")
print(f"\nMissing values:\n{train_df.isnull().sum()}")

churn_counts = train_df['Churn'].value_counts()
churn_pct    = train_df['Churn'].value_counts(normalize=True) * 100
print(f"\nChurn Distribution:")
print(f"  Not Churned: {churn_counts[0]}  ({churn_pct[0]:.1f}%)")
print(f"  Churned:     {churn_counts[1]}  ({churn_pct[1]:.1f}%)")


fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Customer Churn - Exploratory Data Analysis', fontsize=16, fontweight='bold')

colors = ['#2ecc71', '#e74c3c']

axes[0, 0].pie(churn_counts, labels=['Not Churned', 'Churned'],
               autopct='%1.1f%%', colors=colors, startangle=90,
               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0, 0].set_title('Churn Distribution', fontsize=13)

train_df[train_df['Churn'] == 0]['Age'].hist(ax=axes[0, 1], alpha=0.6, color='#2ecc71', label='Not Churned', bins=25)
train_df[train_df['Churn'] == 1]['Age'].hist(ax=axes[0, 1], alpha=0.6, color='#e74c3c', label='Churned', bins=25)
axes[0, 1].set_title('Age Distribution by Churn', fontsize=13)
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend()

support_churn = train_df.groupby('Support Calls')['Churn'].mean().reset_index()
axes[0, 2].bar(support_churn['Support Calls'], support_churn['Churn'], color='#3498db', edgecolor='white')
axes[0, 2].set_title('Churn Rate by Support Calls', fontsize=13)
axes[0, 2].set_xlabel('Support Calls')
axes[0, 2].set_ylabel('Churn Rate')

sub_churn = train_df.groupby('Subscription Type')['Churn'].mean().reset_index()
axes[1, 0].bar(sub_churn['Subscription Type'], sub_churn['Churn'],
               color=['#9b59b6', '#e67e22', '#1abc9c'], edgecolor='white')
axes[1, 0].set_title('Churn Rate by Subscription Type', fontsize=13)
axes[1, 0].set_xlabel('Subscription Type')
axes[1, 0].set_ylabel('Churn Rate')

con_churn = train_df.groupby('Contract Length')['Churn'].mean().reset_index()
axes[1, 1].bar(con_churn['Contract Length'], con_churn['Churn'],
               color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='white')
axes[1, 1].set_title('Churn Rate by Contract Length', fontsize=13)
axes[1, 1].set_xlabel('Contract Length')
axes[1, 1].set_ylabel('Churn Rate')

num_cols = train_df.select_dtypes(include=[np.number]).drop(columns=['CustomerID'])
corr = num_cols.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, ax=axes[1, 2], annot=True, fmt='.2f', cmap='RdYlGn',
            mask=mask, linewidths=0.5, annot_kws={'size': 7})
axes[1, 2].set_title('Feature Correlation Heatmap', fontsize=13)

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nEDA plots saved.")


def preprocess(df):
    df = df.copy()
    df.drop(columns=['CustomerID'], inplace=True)
    le = LabelEncoder()
    for col in ['Gender', 'Subscription Type', 'Contract Length']:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

train_df = train_df.dropna()
test_df  = test_df.dropna()

train_clean = preprocess(train_df)
test_clean  = preprocess(test_df)

X_train = train_clean.drop(columns=['Churn'])
y_train = train_clean['Churn'].astype(int)
X_test  = test_clean.drop(columns=['Churn'])
y_test  = test_clean['Churn'].astype(int)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weight_dict = {0: cw[0], 1: cw[1]}

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Test samples:     {X_test.shape[0]}")
print(f"Class weights:    {class_weight_dict}")


models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=500, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(class_weight='balanced', max_depth=8, random_state=42),
    'Random Forest':       RandomForestClassifier(class_weight='balanced', n_estimators=150,
                                                   max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                       learning_rate=0.1, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
        'Precision': prec, 'Recall': rec, 'F1': f1, 'AUC': auc
    }

    print(f"\n{name}")
    print(f"  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")


metrics_df = pd.DataFrame({
    k: {m: v for m, v in v.items() if m not in ('model', 'y_pred', 'y_prob')}
    for k, v in results.items()
}).T

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Comparison', fontsize=15, fontweight='bold')

x = np.arange(len(metrics_df))
width = 0.2
metric_names  = ['Precision', 'Recall', 'F1', 'AUC']
metric_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

for i, (metric, color) in enumerate(zip(metric_names, metric_colors)):
    axes[0].bar(x + i * width, metrics_df[metric], width, label=metric, color=color, edgecolor='white')

axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(metrics_df.index, rotation=15, ha='right')
axes[0].set_ylim(0, 1.1)
axes[0].set_title('Metrics by Model')
axes[0].legend(loc='lower right')
axes[0].set_ylabel('Score')

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={res['AUC']:.3f})", linewidth=2)
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1].set_title('ROC Curves')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(fontsize=8)

for name, res in results.items():
    prec_c, rec_c, _ = precision_recall_curve(y_test, res['y_prob'])
    ap = average_precision_score(y_test, res['y_prob'])
    axes[2].plot(rec_c, prec_c, label=f"{name} (AP={ap:.3f})", linewidth=2)
axes[2].set_title('Precision-Recall Curves')
axes[2].set_xlabel('Recall')
axes[2].set_ylabel('Precision')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nModel comparison plot saved.")


best_name = max(results, key=lambda x: results[x]['F1'])
best = results[best_name]

print(f"\nBest Model: {best_name}  (F1 = {best['F1']:.4f})")
print(classification_report(y_test, best['y_pred'], target_names=['Not Churned', 'Churned']))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Best Model: {best_name}', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

if hasattr(best['model'], 'feature_importances_'):
    fi = pd.Series(best['model'].feature_importances_, index=X_train.columns).sort_values(ascending=True)
    fi.plot(kind='barh', ax=axes[1], color='#3498db', edgecolor='white')
    axes[1].set_title('Feature Importances')
    axes[1].set_xlabel('Importance')
else:
    coef = pd.Series(np.abs(best['model'].coef_[0]), index=X_train.columns).sort_values(ascending=True)
    coef.plot(kind='barh', ax=axes[1], color='#e74c3c', edgecolor='white')
    axes[1].set_title('Feature Coefficients')
    axes[1].set_xlabel('Absolute Coefficient')

plt.tight_layout()
plt.savefig('best_model_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Best model analysis saved.")


print("\n--- Final Results ---")
print(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
print("-" * 58)
for name, res in results.items():
    print(f"{name:<25} {res['Precision']:>10.4f} {res['Recall']:>10.4f} {res['F1']:>10.4f} {res['AUC']:>10.4f}")

print(f"\nBest: {best_name}")
print(f"  Precision : {best['Precision']:.4f}")
print(f"  Recall    : {best['Recall']:.4f}")
print(f"  F1 Score  : {best['F1']:.4f}")
print(f"  AUC-ROC   : {best['AUC']:.4f}")
