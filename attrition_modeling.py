import pandas as pd
import numpy as np
import os
from data_cleaning import clean_data
from feature_transformers import OrdinalTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, RocCurveDisplay
import matplotlib.pyplot as plt


# load the data
df = clean_data()

# featuers + target
X = df.drop(columns=['Attrition'])
y = df['Attrition']

# numeric columns and categorical columns
ordinal_features = [
        'EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction', 
        'WorkLifeBalance', 'SelfRating', 'ManagerRating', 'Education', 'StockOptionLevel', 'TrainingOpportunitiesWithinYear',
        'TrainingOpportunitiesTaken'
    ]
categorical_features = X.select_dtypes(include=["object", "category"]).columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns

# drop ordinal_features from other feature lists
categorical_features = [col for col in categorical_features if col not in ordinal_features]
numeric_features = [col for col in numeric_features if col not in ordinal_features]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("ordinal", OrdinalTransformer(ordinal_variables=ordinal_features), ordinal_features),
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)

# train full models
rf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=12, class_weight="balanced"))
])
rf.fit(X_train, y_train)

log_reg = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)
y_proba_log = log_reg.predict_proba(X_test)[:,1]

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:,1]

print("Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_log))
print(classification_report(y_test, y_pred_log))

print("Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))
print(classification_report(y_test, y_pred_rf))

result = permutation_importance(rf, X_train, y_train, n_repeats=10, random_state=42, scoring="roc_auc")
importances = pd.Series(result.importances_mean, index=X_train.columns).sort_values(ascending=False)
top_features = importances.head(10).index.tolist()
print("Top 10 features:\n", top_features)

# limited dataset
X_train_lim = X_train[top_features]
X_test_lim = X_test[top_features]

# limit feature categories for preprocessing
ordinal_lim = [f for f in top_features if f in ordinal_features]
categorical_lim = [f for f in top_features if f in categorical_features]
numeric_lim = [f for f in top_features if f in numeric_features]

preprocessor_lim = ColumnTransformer(
    transformers=[
        ("ordinal", OrdinalTransformer(ordinal_variables=ordinal_lim), ordinal_lim),
        ("num", StandardScaler(), numeric_lim),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_lim)
    ]
)

# train limited models
rf_lim = Pipeline([
    ("preprocessor", preprocessor_lim),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=12, class_weight="balanced"))
])
rf_lim.fit(X_train_lim, y_train)

log_reg_lim = Pipeline([
    ("preprocessor", preprocessor_lim),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
log_reg_lim.fit(X_train_lim, y_train)

# evaluate
y_pred_log_lim = log_reg_lim.predict(X_test)
y_proba_log_lim = log_reg_lim.predict_proba(X_test)[:,1]

y_pred_rf_lim = rf.predict(X_test)
y_proba_rf_lim = rf.predict_proba(X_test)[:,1]

print("Limited Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log_lim))
print("Precision:", precision_score(y_test, y_pred_log_lim))
print("Recall:", recall_score(y_test, y_pred_log_lim))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_log_lim))
print(classification_report(y_test, y_pred_log_lim))

print("Limited Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_log))
print(classification_report(y_test, y_pred_log))


# plot roc curves
os.makedirs("plots/model", exist_ok=True)
ax = plt.gca()

y_proba_rf_full = rf.predict_proba(X_test)[:,1]
y_proba_log_full = log_reg.predict_proba(X_test)[:,1]

RocCurveDisplay.from_predictions(y_test, y_proba_rf_full, name="RF Full", ax=ax)
RocCurveDisplay.from_predictions(y_test, y_proba_log_full, name="Logistic Full", ax=ax)
RocCurveDisplay.from_predictions(y_test, y_proba_rf_lim, name="RF Limited", ax=ax)
RocCurveDisplay.from_predictions(y_test, y_proba_log_lim, name="Logistic Limited", ax=ax)


plt.plot([0,1],[0,1],"k--")
plt.title("ROC Curve: Full vs Limited Models")
plt.savefig("Plots/model/roc_curve_comparison.png", dpi=300, bbox_inches="tight")
plt.show()