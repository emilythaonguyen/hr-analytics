import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.base import clone
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
    """
    Build a preprocessing pipeline for numeric and categorical features.

    Using scikit-learn ColumnTransformer that applies:
        - StandardScaler() to numeric features
        - OneHotEncoder() to categorical features
    
    Parameters
    ----------
    X: pd.DataFrame
        The input dataset containing both numeric and categorical columns
    numeric_features : list of str
        Column names in 'X' that are numeric
    categorical_features: list of str
        Column names in 'X' that are categorical

    Return
    ------
    ColumnTransformer
        A fitted ColumnTransformer that can be used in a modeling pipeline.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

def train_model(X_train, y_train, model):
    """
    Build and train a machine learning pipeline.

    Constructs a scikit-learn Pipeline that applies the preprocessing step
    stored in 'X_train.preprocessor', followed by fitting the specified
    classicfication model on the training data.

    Parameters
    ----------
    X_train: DataWrapper
        Training features wrapped in a custom DataWarapper object that contains 
        both the feature data ('.data') and the preprocessing transformer ('.preprocessor')
    y_train : pd.Series or np.ndarray
        Target labels for training
    model: estimator
        A scikit-learn compatible clasifier (e.g., LogisticRegression, RandomForestClassifier)
    
    Returns
    -------
    pipe: Pipeline
        A fitted scikit-learn Pipeline consisting of preprocssing and the trained model.
    """
    pipe = Pipeline([
        ("preprocessor", X_train.preprocessor),
        ("classifier", model)
    ])
    pipe.fit(X_train.data, y_train)
    return pipe

def evaluate_model(model, X_test, y_test, name="Model"):
    """
    Evaluate a trained classification model on test data.

    Generates predictions and predicted probabilities for the given test set, 
    computes evaluation metrics, prints a detailed classification report, 
    and visualizes a confusion matrix.

    Parameters
    ----------
    model : Pipeline or estimator
        A trained scikit-learn compatible model or pipeline with 'predict' 
        and 'predict_proba' methods.
    X_test : pd.DataFrame or np.ndarray
        Test feature set.
    y_test : pd.Series or np.ndarray
        True labels for the test set.
    name : str, optional (default="Model")
        A name used in printed output and plot titles.

    Returns
    -------
    y_proba : np.ndarray
        Predicted probabilities for the positive class.
    """
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
    """
    Perform cross-validation on a classification model pipeline.

    Uses stratified K-fold cross-validation to evaluate the model and prints
    the mean and standard ddeviation of the scores across folds.

    Parameters
    ----------
    model : Pipieline or estimator
        A scikit-learn compatible model or pipeline with 'fit' and 'predict'
        methods.
    X : pd.DataFrame or np.ndarray
        Feature dataset.
    y : pd.Series or np.ndarray
        Target labels.
    cv : int, optional (default=5)
        Number of cross-validation folds
    scoring : str, default="roc_auc"

    Returns
    -------
    scores: np.ndarray
        Array of cross-validation scores for each fold.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    print(f"{scoring} ({cv}-fold CV): mean={scores.mean():.3f}, std={scores.std():.3f}")

def cross_validate_ensemble(rf, log_reg, X, y, cv=5):
    """
    Perform cross-validation for an ensemble of Random Forest and Logistic Regression

    For each fold, both models are trained independently on the training set,
    than their predicted probabilities are avearged to form an ensemble prediction.
    The ROC-AUC score is computed for the ensemble on the validation set.

    Parameters
    ----------
    rf : estimator
        A scikit-laern Random Forest classifier
    log_reg : estimator
        A scikit-learn Logisitic Regression classifier
    X : pd.DataFrame
    y : pd.Series
        Target labels
    cv : int, default = 5
        Number of cross-validation folds.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        rf_fold = clone(rf)
        log_reg_fold = clone(log_reg)
        rf_fold.fit(X_train, y_train)
        log_reg_fold.fit(X_train, y_train)
        
        y_proba_rf = rf_fold.predict_proba(X_val)[:, 1]
        y_proba_log = log_reg_fold.predict_proba(X_val)[:, 1]
        y_proba_ensemble = (y_proba_rf + y_proba_log) / 2
        
        auc = roc_auc_score(y_val, y_proba_ensemble)
        aucs.append(auc)
    print(f"Ensemble ROC-AUC ({cv}-fold CV): mean={np.mean(aucs):.3f}, std={np.std(aucs):.3f}")

# building our models!
def run_pipeline(dataset_name="hr_only"):
    """
    Run the full HR attrition modeling pipeline for a given dataset

    Parameters
    ----------
    dataset_name : str, default="hr_only"
        Name of the dataset to load and process
    """
    # load and clean data
    df = clean_data(dataset_name)
    if 'Education' or 'StockOptionLevel' in df.columns:
        df['Education'] = df['Education'].cat.codes  # convert ordinal to int
        df['StockOptionLevel'] = df['StockOptionLevel'].cat.codes

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
    
    print("Cross-validating Ensemble (Full Models)...")
    cross_validate_ensemble(rf, log_reg, X_train.data, y_train, cv=5)

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