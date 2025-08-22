import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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
            if pd.api.types.is_categorical_dtype(X[var]) and X[var].cat.ordered:
                X[var] = X[var].catcodes
            else:
                X[var] = X[var].astype(int)
        
        return X