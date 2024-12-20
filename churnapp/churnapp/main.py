from typing import Union


from typing import Union


from fastapi import FastAPI
from pydantic import BaseModel
import dill
import pandas as pd
import numpy as np

app = FastAPI()

with open('final_xgboost_model.pkl','rb')as f:
    reloaded_model = dill.load(f)

class payload(BaseModel):
    Age: float
    Balance: float
    NumOfProducts: int
    EstimatedSalary: float
    Gender: str
    Age_Balance_Ratio: float
    Credit_Age_Ratio: float
    Balance_Products_Ratio: float

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None,
              x: Union[str, None] = None):
    return {"item_id": item_id, "q": q,"x": x}


@app.post("/predict")
def predict (payload:Payload):
    df = pd.DataFrame([payload.model_dump.values()],columns = payload.model_dump().keys())

    y_hat = reloaded_model.predict(df)
    return{"prediction":y_hat[0]}