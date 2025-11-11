import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pydantic import BaseModel
from catboost import CatBoostClassifier, Pool

DATA_PATH = "data"
MODEL_FILENAME = "purchase_model_v3.pkl"

class SectorEnum(str, Enum):
    NONSAUDI_PRIVT = "NONSAUDI_PRIVT"
    PRIVT = "PRIVT"
    GOV_NONMILITARY = "GOV-NONMILITARY"
    GOV_MILITARY = "GOV-MILITARY"
    RETMT = "RETMT"
    NONSAUDI_GOV_NONMILITARY = "NONSAUDI_GOV-NONMILITARY"

class GenderEnum(str, Enum):
    M = "M"
    F = "F"

class InputData(BaseModel):
    age: float
    simah_score: float
    sector: SectorEnum
    income: float
    dbr_ofo_amt: float
    Gender: GenderEnum
    requested_loan_amount: float
    basic_wage: float
    housing_allowance: float
    svr_jdg_count: float
    svr_defaultamount: float
    dbr_mal_bmo: float
    days_in_stage: float
    date_of_joining: Optional[datetime] = None
    CreatedOn: Optional[datetime] = None

class ScoreModel:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(DATA_PATH, MODEL_FILENAME)
        with open(model_path, "rb") as f:
            self.model: CatBoostClassifier = pickle.load(f)
        self.numeric_cols = [
            "age", "simah_score", "income", "dbr_ofo_amt",
            "requested_loan_amount", "basic_wage", "housing_allowance",
            "svr_jdg_count", "svr_defaultamount", "dbr_mal_bmo", "days_in_stage"
        ]
        self.cat_features = ["sector", "Gender"]

    def preprocess_data(self, input_data: InputData):
        data = input_data.dict()

        # Если days_in_stage уже есть (и не None/NaN) — не пересчитываем
        if data.get("days_in_stage") is not None and not np.isnan(data.get("days_in_stage")):
            # Удаляем даты, если они не нужны
            data.pop("date_of_joining", None)
            data.pop("CreatedOn", None)
        else:
            # Нет days_in_stage — рассчитываем по датам
            if not data.get("date_of_joining"):
                data["date_of_joining"] = datetime(2023, 1, 1)
            if not data.get("CreatedOn"):
                data["CreatedOn"] = datetime.now()
            date_join = data.pop("date_of_joining")
            created_on = data.pop("CreatedOn")
            data["days_in_stage"] = (created_on - date_join).days

        return pd.DataFrame([data])

    def predict(self, input_data: InputData, threshold: float = 0.5):
        data_df = self.preprocess_data(input_data)
        pool = Pool(data_df, cat_features=self.cat_features)
        proba = float(self.model.predict_proba(pool)[0, 1])
        decision = "Approve" if proba < threshold else "Reject"
        shap_values_all = self.model.get_feature_importance(pool, type="ShapValues")[0]
        shap_values = shap_values_all[:-1]
        shap_output = [{"feature": f, "value": float(v)} for f, v in zip(data_df.columns, shap_values)]
        return data_df, {
            "probability_of_default": proba,
            "decision": decision,
            "shap_values": shap_output
        }
