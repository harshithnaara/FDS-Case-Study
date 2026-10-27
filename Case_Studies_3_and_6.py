
# Fundamentals of Data Science - Case Studies


# Libraries and Settings

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)

os.makedirs('figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'text.color': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.titleweight': 'bold',
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 100,
})

def save_show(name):
    plt.tight_layout()
    plt.savefig('figures/' + name + '.png', dpi=200, facecolor='white')
    plt.show()

def plot_confusion(ax, cm, title):
    ax.imshow(cm, cmap='Greys', vmin=0, vmax=cm.max() * 1.6)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=13,
                    color='white' if cm[i, j] > cm.max() * 0.6 else 'black')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted No', 'Predicted Yes'])
    ax.set_yticklabels(['Actual No', 'Actual Yes'])
    ax.set_title(title)


# Case Study 3: Logistic Regression on the Titanic Dataset


# 1. Data Exploration

titanic = pd.read_excel('data/titanic.xlsx')
display(titanic.head(10))
print(titanic.shape)
print(titanic.dtypes)

display(titanic.describe())
print(titanic.isnull().sum())

survival_counts = titanic['Survived'].value_counts().sort_index()
print(survival_counts)
print((survival_counts / len(titanic) * 100).round(2))

plt.figure(figsize=(5, 4))
bars = plt.bar(['Did not survive', 'Survived'], survival_counts.values, color='lightgrey', edgecolor='black', hatch='//')
for b in bars:
    plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 8, int(b.get_height()), ha='center')
plt.title('Survival Count of Passengers')
plt.xlabel('Survival Status')
plt.ylabel('Number of Passengers')
save_show('cs3_survival_count')

sex_rate = titanic.groupby('Sex')['Survived'].mean() * 100
class_rate = titanic.groupby('Pclass')['Survived'].mean() * 100
print(sex_rate.round(2))
print(class_rate.round(2))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
b1 = axes[0].bar(sex_rate.index.str.title(), sex_rate.values, color='lightgrey', edgecolor='black', hatch='//')
axes[0].set_title('Survival Rate by Gender')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Survival Rate (%)')
b2 = axes[1].bar(['1st', '2nd', '3rd'], class_rate.values, color='lightgrey', edgecolor='black', hatch='\\\\')
axes[1].set_title('Survival Rate by Passenger Class')
axes[1].set_xlabel('Passenger Class')
axes[1].set_ylabel('Survival Rate (%)')
for ax, bars in zip(axes, [b1, b2]):
    ax.set_ylim(0, 100)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f'{b.get_height():.1f}%', ha='center')
save_show('cs3_survival_gender_class')

plt.figure(figsize=(7, 4))
plt.hist([titanic.loc[titanic['Survived'] == 0, 'Age'].dropna(),
          titanic.loc[titanic['Survived'] == 1, 'Age'].dropna()],
         bins=15, color=['dimgrey', 'white'], edgecolor='black', label=['Did not survive', 'Survived'])
plt.title('Age Distribution of Passengers by Survival')
plt.xlabel('Age (years)')
plt.ylabel('Number of Passengers')
plt.legend()
save_show('cs3_age_distribution')

fare_groups = [titanic.loc[titanic['Survived'] == s, 'Fare'] for s in [0, 1]]
plt.figure(figsize=(6, 4))
plt.boxplot(fare_groups, tick_labels=['Did not survive', 'Survived'], showfliers=False,
            medianprops={'color': 'black'}, boxprops={'color': 'black'},
            whiskerprops={'color': 'black'}, capprops={'color': 'black'})
plt.title('Ticket Fare by Survival Status')
plt.xlabel('Survival Status')
plt.ylabel('Fare')
save_show('cs3_fare_boxplot')

numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr = titanic[numeric_cols].corr()
print(corr.round(2))

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.imshow(corr, cmap='Greys', vmin=-1, vmax=1)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
ax.set_yticklabels(numeric_cols)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        v = corr.iloc[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', color='white' if abs(v) > 0.55 else 'black')
ax.set_title('Correlation Matrix of Numerical Features')
save_show('cs3_correlation')


# 2. Data Preprocessing

data = titanic.copy()
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True, dtype=int)
print(data.isnull().sum().sum())
display(data.head())


# 3. Model Building

X = data.drop(columns='Survived')
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(X_train.shape, X_test.shape)

logit = LogisticRegression(max_iter=1000)
logit.fit(X_train, y_train)


# 4. Model Evaluation

pred = logit.predict(X_test)
prob = logit.predict_proba(X_test)[:, 1]

cs3_metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Value': [accuracy_score(y_test, pred), precision_score(y_test, pred),
              recall_score(y_test, pred), f1_score(y_test, pred), roc_auc_score(y_test, prob)]
}).round(4)
display(cs3_metrics)
cs3_metrics.to_csv('outputs/cs3_metrics.csv', index=False)
print('Training accuracy:', round(logit.score(X_train, y_train), 4))
print(classification_report(y_test, pred))

cm = confusion_matrix(y_test, pred)
print(cm)
fig, ax = plt.subplots(figsize=(5, 4.5))
plot_confusion(ax, cm, 'Confusion Matrix - Logistic Regression')
ax.set_xlabel('Predicted Class')
ax.set_ylabel('Actual Class')
save_show('cs3_confusion_matrix')
pd.DataFrame(cm, index=['Actual No', 'Actual Yes'], columns=['Predicted No', 'Predicted Yes']).to_csv('outputs/cs3_confusion.csv')

fpr, tpr, _ = roc_curve(y_test, prob)
plt.figure(figsize=(5.5, 5))
plt.plot(fpr, tpr, color='black', linewidth=2, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, prob):.3f})')
plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1, label='Random Guess')
plt.title('ROC Curve - Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
save_show('cs3_roc_curve')


# 5. Interpretation of Coefficients

coef_table = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logit.coef_[0],
    'Odds Ratio': np.exp(logit.coef_[0])
}).sort_values('Coefficient', ascending=False).round(4)
display(coef_table)
coef_table.to_csv('outputs/cs3_coefficients.csv', index=False)

ordered = coef_table.sort_values('Coefficient')
plt.figure(figsize=(7, 4.5))
plt.barh(ordered['Feature'], ordered['Coefficient'], color='lightgrey', edgecolor='black', hatch='//')
plt.axvline(0, color='black', linewidth=1)
plt.title('Logistic Regression Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
save_show('cs3_coefficients')


# 6. Deployment with Streamlit

joblib.dump({'model': logit, 'columns': list(X.columns), 'age_median': float(titanic['Age'].median())}, 'titanic_model.pkl')


# Case Study 6: Bank Marketing Classification


# 1. Data Understanding and Preprocessing

bank = pd.read_csv('data/bank-full.csv', sep=';')
display(bank.head(10))
print(bank.shape)
print(bank.dtypes)

display(bank.describe())
numerical_cols = bank.select_dtypes(include='number').columns.tolist()
categorical_cols = bank.select_dtypes(exclude='number').columns.drop('y').tolist()
print('Numerical:', numerical_cols)
print('Categorical:', categorical_cols)
print('Target: y')

print(bank.isnull().sum().sum())
unknown_counts = (bank[categorical_cols] == 'unknown').sum()
print(unknown_counts[unknown_counts > 0])
print((unknown_counts[unknown_counts > 0] / len(bank) * 100).round(2))
print('Duplicates:', bank.duplicated().sum())
print(bank['y'].value_counts())

model_data = bank.drop(columns='duration').copy()
for col in ['default', 'housing', 'loan', 'y']:
    model_data[col] = (model_data[col] == 'yes').astype(int)
model_data = pd.get_dummies(model_data, columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome'],
                            drop_first=True, dtype=int)
Xb = model_data.drop(columns='y')
yb = model_data['y']
print(Xb.shape)

Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.2, stratify=yb, random_state=42)

scale_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
scaler = StandardScaler()
Xb_train_s = Xb_train.copy()
Xb_test_s = Xb_test.copy()
Xb_train_s[scale_cols] = scaler.fit_transform(Xb_train[scale_cols])
Xb_test_s[scale_cols] = scaler.transform(Xb_test[scale_cols])
print(Xb_train_s.shape, Xb_test_s.shape)
print(yb_train.mean().round(4), yb_test.mean().round(4))


# 2. Exploratory Data Analysis

target_counts = bank['y'].value_counts()
target_pct = (target_counts / len(bank) * 100).round(2)
print(pd.DataFrame({'Count': target_counts, 'Percentage': target_pct}))

plt.figure(figsize=(5.5, 4.5))
bars = plt.bar(['No', 'Yes'], [target_counts['no'], target_counts['yes']], color='lightgrey', edgecolor='black', hatch='//')
for b, p in zip(bars, [target_pct['no'], target_pct['yes']]):
    plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 500, f'{int(b.get_height())} ({p}%)', ha='center')
plt.title('Distribution of Term Deposit Subscription')
plt.xlabel('Subscribed to Term Deposit')
plt.ylabel('Number of Customers')
plt.ylim(0, 48000)
save_show('cs6_target_distribution')

job_rate = (bank.groupby('job')['y'].apply(lambda s: (s == 'yes').mean() * 100)).sort_values()
print(job_rate.round(2))

plt.figure(figsize=(7, 5))
plt.barh(job_rate.index, job_rate.values, color='lightgrey', edgecolor='black', hatch='//')
plt.title('Subscription Rate by Job Type')
plt.xlabel('Subscription Rate (%)')
plt.ylabel('Job Type')
save_show('cs6_rate_by_job')

month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month_rate = bank.groupby('month')['y'].apply(lambda s: (s == 'yes').mean() * 100).reindex(month_order)
month_count = bank['month'].value_counts().reindex(month_order)
print(pd.DataFrame({'Rate': month_rate.round(2), 'Contacts': month_count}))

plt.figure(figsize=(8, 4.5))
plt.bar([m.title() for m in month_order], month_rate.values, color='lightgrey', edgecolor='black', hatch='\\\\')
plt.title('Subscription Rate by Month of Last Contact')
plt.xlabel('Month')
plt.ylabel('Subscription Rate (%)')
save_show('cs6_rate_by_month')

pout_rate = bank.groupby('poutcome')['y'].apply(lambda s: (s == 'yes').mean() * 100).sort_values()
print(pout_rate.round(2))

plt.figure(figsize=(6, 4.5))
plt.bar([p.title() for p in pout_rate.index], pout_rate.values, color='lightgrey', edgecolor='black', hatch='xx')
plt.title('Subscription Rate by Previous Campaign Outcome')
plt.xlabel('Previous Campaign Outcome')
plt.ylabel('Subscription Rate (%)')
save_show('cs6_rate_by_poutcome')

print(bank.groupby('y')[['age', 'balance', 'campaign', 'previous']].median())

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
for ax, col, label in zip(axes, ['age', 'balance', 'campaign'], ['Age (years)', 'Balance (euros)', 'Number of Contacts']):
    ax.boxplot([bank.loc[bank['y'] == 'no', col], bank.loc[bank['y'] == 'yes', col]],
               tick_labels=['No', 'Yes'], showfliers=False,
               medianprops={'color': 'black'}, boxprops={'color': 'black'},
               whiskerprops={'color': 'black'}, capprops={'color': 'black'})
    ax.set_title(label.split(' (')[0] + ' vs Subscription')
    ax.set_xlabel('Subscribed')
    ax.set_ylabel(label)
save_show('cs6_numeric_boxplots')


# 3. Classification Model Development

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
}

rows = []
predictions = {}
probabilities = {}
matrices = {}
for name, model in models.items():
    model.fit(Xb_train_s, yb_train)
    pred_b = model.predict(Xb_test_s)
    prob_b = model.predict_proba(Xb_test_s)[:, 1]
    predictions[name] = pred_b
    probabilities[name] = prob_b
    matrices[name] = confusion_matrix(yb_test, pred_b)
    rows.append({
        'Model': name,
        'Accuracy': accuracy_score(yb_test, pred_b),
        'Precision': precision_score(yb_test, pred_b, zero_division=0),
        'Recall': recall_score(yb_test, pred_b),
        'F1-Score': f1_score(yb_test, pred_b),
        'ROC-AUC': roc_auc_score(yb_test, prob_b),
    })

results = pd.DataFrame(rows).set_index('Model').round(4)
display(results)
results.to_csv('outputs/cs6_metrics.csv')


# 4. Model Performance Comparison and Visualization

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
patterns = ['', '//', '..', 'xx']
x = np.arange(len(results))
width = 0.2

plt.figure(figsize=(10, 5))
for i, (metric, pattern) in enumerate(zip(metric_names, patterns)):
    plt.bar(x + (i - 1.5) * width, results[metric], width, label=metric,
            color='white' if i % 2 == 0 else 'lightgrey', edgecolor='black', hatch=pattern)
plt.xticks(x, results.index)
plt.ylim(0, 1.05)
plt.title('Performance Comparison of Classification Models')
plt.xlabel('Model')
plt.ylabel('Score')
plt.legend(loc='upper center', ncol=4)
save_show('cs6_model_comparison')

fig, axes = plt.subplots(2, 2, figsize=(9, 8))
for ax, (name, cm_b) in zip(axes.ravel(), matrices.items()):
    plot_confusion(ax, cm_b, name)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')
save_show('cs6_confusion_matrices')
pd.DataFrame({name: m.ravel() for name, m in matrices.items()}, index=['TN', 'FP', 'FN', 'TP']).T.to_csv('outputs/cs6_confusion.csv')
print(pd.DataFrame({name: m.ravel() for name, m in matrices.items()}, index=['TN', 'FP', 'FN', 'TP']).T)

styles = {'Logistic Regression': '-', 'Decision Tree': '--', 'KNN': ':', 'Random Forest': '-.'}
widths = {'Logistic Regression': 2, 'Decision Tree': 2, 'KNN': 2.5, 'Random Forest': 2}
plt.figure(figsize=(6.5, 5.5))
for name in models:
    fpr_b, tpr_b, _ = roc_curve(yb_test, probabilities[name])
    plt.plot(fpr_b, tpr_b, color='black', linestyle=styles[name], linewidth=widths[name],
             label=f"{name} (AUC = {results.loc[name, 'ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], color='grey', linewidth=1, label='Random Guess')
plt.title('ROC Curves of the Four Classifiers')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
save_show('cs6_roc_curves')


# 5. Interpretation and Business Recommendations

best_model = results['ROC-AUC'].idxmax()
print('Best model:', best_model)
print(classification_report(yb_test, predictions[best_model], target_names=['No', 'Yes']))

importance = pd.Series(models[best_model].feature_importances_, index=Xb_train_s.columns).sort_values(ascending=False).head(10)
display(importance.round(4))
importance.round(4).to_csv('outputs/cs6_importance.csv', header=['Importance'])

plt.figure(figsize=(7, 5))
plt.barh(importance.index[::-1], importance.values[::-1], color='lightgrey', edgecolor='black', hatch='//')
plt.title('Top 10 Important Features - ' + best_model)
plt.xlabel('Importance Score')
plt.ylabel('Feature')
save_show('cs6_feature_importance')

ranking = pd.DataFrame({'actual': yb_test.values, 'score': probabilities[best_model]}).sort_values('score', ascending=False).reset_index(drop=True)
total_yes = ranking['actual'].sum()
gain_rows = []
for share in [0.10, 0.20, 0.30, 0.50]:
    top = ranking.head(int(len(ranking) * share))
    gain_rows.append({
        'Customers Contacted (%)': int(share * 100),
        'Subscribers Found': int(top['actual'].sum()),
        'Share of All Subscribers (%)': round(top['actual'].sum() / total_yes * 100, 2),
        'Subscription Rate (%)': round(top['actual'].mean() * 100, 2),
    })
gain = pd.DataFrame(gain_rows)
display(gain)
gain.to_csv('outputs/cs6_gain.csv', index=False)
