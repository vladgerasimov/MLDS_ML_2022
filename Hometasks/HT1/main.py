import pickle
from joblib import load
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import json
from custom_transformer import CustomNumericTransformer

app = FastAPI()
# pipeline = None


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


pipeline = load('pipeline.pkl')


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """
    Cannot inference model directly on single object due to usage of df's method 'apply' in pipeline
    """
    df = pd.DataFrame(item.dict(), index=[0])
    return pipeline.predict(df)[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    for idx, item in enumerate(items):
        items[idx] = item.dict()
    df = pd.DataFrame(items)
    return pipeline.predict(df).tolist()


@app.post("/predict_items_csv")
def predict_items(file: UploadFile = File(...)) -> FileResponse:
    df = pd.read_csv(file.file, skipinitialspace=True)
    df['prediction'] = pipeline.predict(df)
    df.to_csv('tmp_df.csv')
    return FileResponse('tmp_df.csv')
