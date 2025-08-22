import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    classification_report, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline



from data_cleaning import clean_data
from bivariate import nparametric_tests

# helper functions
def build_preprocessor(X, numeric_features, categorical_features):
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

def train_model(X_train, y_train, model):
    pipe = Pipeline([
        ("preprocessor", X_train.preprocessor),
        ("classifier", model)
    ])
    pipe.fit(X_train.data, y_train)
    return pipe

def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[1,0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Leave","Stay"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {name}")
    plt.show()
    
    return y_proba

def cross_validate_pipeline(model, X, y, cv=5, scoring="roc_auc"):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    print(f"{scoring} ({cv}-fold CV): mean={scores.mean():.3f}, std={scores.std():.3f}")

# building our models!
def run_pipeline(dataset_name="hr_only"):
    # load and clean data
    df = clean_data(dataset_name)
    if 'Education' in df.columns:
        df['Education'] = df['Education'].cat.codes  # convert ordinal to int
    
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    preprocessor = build_preprocessor(X, numeric_features, categorical_features)
    
    preprocessor = build_preprocessor(X, numeric_features, categorical_features)
    
    X_train_data, X_test_data, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Top features
    results = nparametric_tests(df)
    results_df = pd.DataFrame(results)
    sig_vars = results_df.sort_values("p-value").head(10)['Variable'].tolist()
    print(f"Top 10 features: {sig_vars}")
    
    X_train_lim_data = X_train_data[sig_vars]
    X_test_lim_data = X_test_data[sig_vars]
    
    numeric_lim = [f for f in sig_vars if f in numeric_features]
    categorical_lim = [f for f in sig_vars if f in categorical_features]
    preprocessor_lim = build_preprocessor(X_train_lim_data, numeric_lim, categorical_lim)
    
    # wrap preprocessor and data
    class DataWrapper:
        def __init__(self, data, preprocessor):
            self.data = data
            self.preprocessor = preprocessor
    
    X_train = DataWrapper(X_train_data, preprocessor)
    X_test = DataWrapper(X_test_data, preprocessor)
    X_train_lim = DataWrapper(X_train_lim_data, preprocessor_lim)
    X_test_lim = DataWrapper(X_test_lim_data, preprocessor_lim)
    
    # SMOTE object
    smote = SMOTE(random_state=42)
    
    # full models
    rf = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", smote),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    rf.fit(X_train.data, y_train)
    
    log_reg = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", smote),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    log_reg.fit(X_train.data, y_train)
    
    # cross-validate full models
    print("Cross-validating Random Forest")
    cross_validate_pipeline(rf, X_train.data, y_train, cv=5, scoring="roc_auc")
    print("Cross-validating Logistic Regression")
    cross_validate_pipeline(log_reg, X_train.data, y_train, cv=5, scoring="roc_auc")
    
    y_proba_rf = evaluate_model(rf, X_test.data, y_test, name="Random Forest + SMOTE")
    y_proba_log = evaluate_model(log_reg, X_test.data, y_test, name="Logistic Regression + SMOTE")
    
    # limited models
    rf_lim = ImbPipeline([
        ("preprocessor", preprocessor_lim),
        ("smote", smote),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    rf_lim.fit(X_train_lim.data, y_train)
    
    log_reg_lim = ImbPipeline([
        ("preprocessor", preprocessor_lim),
        ("smote", smote),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    log_reg_lim.fit(X_train_lim.data, y_train)
    
    # cross-validate limited models
    print("Cross-validating Random Forest (Limited)...")
    cross_validate_pipeline(rf_lim, X_train_lim.data, y_train, cv=5, scoring="roc_auc")
    print("Cross-validating Logistic Regression (Limited)...")
    cross_validate_pipeline(log_reg_lim, X_train_lim.data, y_train, cv=5, scoring="roc_auc")

    y_proba_rf_lim = evaluate_model(rf_lim, X_test_lim.data, y_test, name="Random Forest (Limited) + SMOTE")
    y_proba_log_lim = evaluate_model(log_reg_lim, X_test_lim.data, y_test, name="Logistic Regression (Limited) + SMOTE")
    
    # ensemble
    y_proba_ensemble = (y_proba_rf + y_proba_log) / 2
    y_pred_ensemble = (y_proba_ensemble >= 0.5).astype(int)
    
    print("--- Ensemble (Full Models + SMOTE) ---")
    print("Accuracy:", accuracy_score(y_test, y_pred_ensemble))
    print("Precision:", precision_score(y_test, y_pred_ensemble))
    print("Recall:", recall_score(y_test, y_pred_ensemble))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba_ensemble))
    print(classification_report(y_test, y_pred_ensemble))
    
    cm = confusion_matrix(y_test, y_pred_ensemble, labels=[1,0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Leave","Stay"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: Ensemble + SMOTE")
    plt.show()

    # roc curves
    os.makedirs("plots/model", exist_ok=True)
    ax = plt.gca()
    RocCurveDisplay.from_predictions(y_test, y_proba_rf, name="RF Full", ax=ax)
    RocCurveDisplay.from_predictions(y_test, y_proba_log, name="Logistic Full", ax=ax)
    RocCurveDisplay.from_predictions(y_test, y_proba_rf_lim, name="RF Limited", ax=ax)
    RocCurveDisplay.from_predictions(y_test, y_proba_log_lim, name="Logistic Limited", ax=ax)
    RocCurveDisplay.from_predictions(y_test, y_proba_ensemble, name='Ensemble', ax=ax)
    plt.plot([0,1],[0,1], "k--")
    plt.title("ROC Curve: Full vs Limited Models")
    plt.savefig("plots/model/roc_curve_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    run_pipeline("hr_only")


# # featuers + target
# X = df.drop(columns=['Attrition'])
# y = df['Attrition']

# categorical_features = X.select_dtypes(include=["object", "category"]).columns
# numeric_features = X.select_dtypes(include=["int64", "float64"]).columns


# # full model using only hr data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), numeric_features),
#         ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
#     ]
# )

# # pipelines
# rf = Pipeline([
#     ("preprocessor", preprocessor),
#     ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
# ])

# log_reg = Pipeline([
#     ("preprocessor", preprocessor),
#     ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
# ])

# rf.fit(X_train, y_train)
# log_reg.fit(X_train, y_train)

# def evaluate_model(model, X_test, y_test, model_name="Model"):
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:,1]
#     print(f"{model_name}")
#     print('Accuracy:', accuracy_score(y_test, y_pred))
#     print('Precision:', precision_score(y_test, y_pred))
#     print('Recall:', recall_score(y_test, y_pred))
#     print('ROC-AUC:', roc_auc_score(y_test, y_proba))
#     print(classification_report(y_test, y_pred))
#     return y_proba

# y_proba_rf = evaluate_model(rf, X_test, y_test, "Random Forest")
# y_proba_log = evaluate_model(log_reg, X_test, y_test, "Logistic Regression")


# result = permutation_importance(rf, X_train, y_train, n_repeats=10, random_state=42, scoring="roc_auc")
# importances = pd.Series(result.importances_mean, index=X_train.columns).sort_values(ascending=False)
# top_features = importances.head(10).index.tolist()
# print("Top 10 features:\n", top_features)



# # limited dataset
# X_train_lim = X_train[top_features]
# X_test_lim = X_test[top_features]

# # identify numeric and categorical in limited features
# categorical_lim = [f for f in top_features if f in categorical_features]
# numeric_lim = [f for f in top_features if f in numeric_features]

# preprocessor_lim = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), numeric_lim),
#         ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_lim)
#     ]
# )

# # limited pipeline
# rf_lim = Pipeline([
#     ("preprocessor", preprocessor_lim),
#     ("classifier", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
# ])

# log_reg_lim = Pipeline([
#     ("preprocessor", preprocessor_lim),
#     ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
# ])

# # train
# rf_lim.fit(X_train_lim, y_train)
# log_reg_lim.fit(X_train_lim, y_train)

# # predict & evaluate limited models 
# y_proba_rf_lim = evaluate_model(rf_lim, X_test_lim, y_test, "Random Forest (Limited)")
# y_proba_log_lim = evaluate_model(log_reg_lim, X_test_lim, y_test, "Logistic Regression (Limited)")


# # plot roc curves
# os.makedirs("plots/model", exist_ok=True)
# ax = plt.gca()

# RocCurveDisplay.from_predictions(y_test, y_proba_rf, name="RF", ax=ax)
# RocCurveDisplay.from_predictions(y_test, y_proba_log, name="Logistic", ax=ax)
# RocCurveDisplay.from_predictions(y_test, y_proba_rf_lim, name="RF Limited", ax=ax)
# RocCurveDisplay.from_predictions(y_test, y_proba_log_lim, name="Logistic Limited", ax=ax)

# plt.plot([0,1],[0,1],"k--")
# plt.title("ROC Curve: Full vs Limited Models")
# plt.savefig("Plots/model/roc_curve_comparison.png", dpi=300, bbox_inches="tight")
# plt.show()

# # confusion matrix

# # confusion matrix
# # logistic regression
# y_pred = log_reg.predict(X_test)
# cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Leave", "Stay"])
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix: Logistic Regression (Limited)")
# plt.show()

# # random forest
# y_pred = rf.predict(X_test)
# cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Leave", "Stay"])
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix: Random Forest")
# plt.show()

# # logistic regression lim
# y_pred = log_reg_lim.predict(X_test_lim)
# cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Leave", "Stay"])
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix: Logistic Regression (Limited)")
# plt.show()

# # random forest lim
# y_pred = rf_lim.predict(X_test_lim)
# cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Leave", "Stay"])
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix: Random Forest (Limited)")
# plt.show()
