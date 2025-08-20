import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DateTimeTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms datetime variables before preprocessing:
    - year
    - month
    - day
    """
    def __init__(self, reference_date=None):
        
        self.reference_date = pd.to_datetime(reference_date) if reference_date else None
        self.datetime_variables = []

    def fit(self, X, y=None):
        self.datetime_variables = X.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        if self.reference_date is None and self.datetime_variables:
            self.reference_date = X[self.datetime_variables].max().max()
        return self
    
    def transform(self, X):
        X = X.copy()
        for var in self.datetime_variables:
            # Years
            X[var + "_tenure_years"] = (self.reference_date - X[var]).dt.days / 365.25
            # month and weekday - to be more concise
            X[var + "_month"] = X[var].dt.month
            X[var + "_weekday"] = X[var].dt.weekday
        # Drop original datetime columns
        X = X.drop(columns=self.datetime_variables)
        return X

class OrdinalTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to convert ordinal columns
    - Ordinal columns: change dtype to int
    """
    def __init__(self, ordinal_variables=None):
        self.ordinal_variables = ordinal_variables if ordinal_variables else []
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # ordinal variables
        for var in self.ordinal_variables:
            X[var] = X[var].astype(int)
        
        return X