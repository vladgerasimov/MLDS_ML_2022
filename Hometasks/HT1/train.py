import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump
from custom_transformer import CustomNumericTransformer

numeric_feats = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
categorical_feats = ['fuel', 'seller_type', 'transmission', 'owner']
num_and_cat_feats = ['seats']

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")),
           ("scaler", StandardScaler())]
)

categorical_transformer = OneHotEncoder(drop="first", sparse=False)

num_cat_fransformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")),
           ("encoder", OneHotEncoder(drop="first", sparse=False))]
)

custom_prepocessor = CustomNumericTransformer()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_feats),
        ("cat", categorical_transformer, categorical_feats),
        ("num_cat", num_cat_fransformer, num_and_cat_feats)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(
    steps=[("custom_preprocessor", custom_prepocessor),
           ("preprocessor", preprocessor),
           ("classifier", Ridge(alpha=5))
          ]
)

df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
new_index = df_train.drop('selling_price', axis=1).drop_duplicates(keep='first').index
df_train = df_train.loc[new_index]
y_train = df_train['selling_price']

pipeline.fit(df_train, y_train)

with open('pipeline_upd.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

dump(pipeline, 'pipeline.pkl')
