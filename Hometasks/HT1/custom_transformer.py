import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CustomNumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    @staticmethod
    def drop_units_of_measurment(col: pd.Series) -> pd.Series:
        col = col.apply(lambda x: x.split(' ')[0] if isinstance(x, str) else float(x))
        col.replace({'': np.nan}, inplace=True)
        return col.astype(float)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        for col in ('mileage', 'engine', 'max_power'):
            X_[col] = self.drop_units_of_measurment(X_[col])

        if 'torque' in X_.columns:
            X_.drop('torque', axis=1, inplace=True)

        if 'name' in X_.columns:
            X_.drop('name', axis=1, inplace=True)
        return X_
