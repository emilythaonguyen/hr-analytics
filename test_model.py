import pandas as pd
import numpy as np
import os
from data_cleaning import clean_data
from feature_transformers import DateTimeTransformer, OrdinalTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

df = clean_data()

x = df.drop(columns=['Attrition'])
y = df['Attrition']

print(y.value_counts())

datetime_features = x.select_dtypes(include=['datetime64[ns]']).columns.tolist()
categorical_features = x.select_dtypes(include=["object", "category"]).columns
numeric_features = x.select_dtypes(include=["int64", "float64"]).columns

X = pd.get_dummies(x, drop_first=True)




