import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# telco churn dataset from kaggle
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.shape
df.head()
df.info()

# don't need this, it's just an id
df.drop("customerID", axis=1, inplace=True)

# TotalCharges comes in as object because of empty strings for new customers
# took me a minute to figure out why my model kept crashing lol
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].isnull().sum()  # only 11, just drop them
df.dropna(inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---- quick look at the data before modeling ----
df["Churn"].value_counts(normalize=True)
# about 73/27 split, so accuracy alone won't tell us much, keep that in mind later

plt.figure(figsize=(5, 4))
sns.countplot(x="Churn", data=df)
plt.title("Churn counts")
plt.savefig("churn_counts.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True)
plt.title("tenure vs churn")
plt.savefig("tenure_vs_churn.png")
plt.close()

# encoding categorical stuff - just using LabelEncoder across the board for now,
# I know one-hot is more "correct" for non-ordinal cols but this is faster to get working
cat_cols = df.select_dtypes(include="object").columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- baseline model ----
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)

print("Logistic Regression")
print("accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))

# cross validation on the log reg, just to sanity check the train/test split
# wasn't a fluke. using 5 folds, stratified so the churn ratio stays consistent
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_lr = cross_val_score(log_reg, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
print("logreg CV roc-auc scores:", cv_scores_lr)
print("logreg CV mean:", cv_scores_lr.mean(), "+/-", cv_scores_lr.std())

# ---- random forest ----
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)  # trees don't care about scaling
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest")
print("accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

y_proba_rf = rf.predict_proba(X_test)[:, 1]
print("roc-auc:", roc_auc_score(y_test, y_proba_rf))

# same CV check for the forest
cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc")
print("rf CV roc-auc scores:", cv_scores_rf)
print("rf CV mean:", cv_scores_rf.mean(), "+/-", cv_scores_rf.std())

# CV scores being close to the test set score is a good sign we're not just
# getting lucky/unlucky with one particular split

# ---- feature importance, easy to add and useful for the writeup ----
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind="barh")
plt.title("top 10 features (random forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

print("done - plots saved to working directory")
