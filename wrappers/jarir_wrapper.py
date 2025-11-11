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

# ---------------- ENUM CLASSES ---------------- #
class Nationality(str, Enum):
    Afghanistan = "Afghanistan"
    Albania = "Albania"
    Algeria = "Algeria"
    Andorra = "Andorra"
    Angola = "Angola"
    Antigua_and_Barbuda = "Antigua and Barbuda"
    Argentina = "Argentina"
    Armenia = "Armenia"
    Aruba = "Aruba"
    Australia = "Australia"
    Austria = "Austria"
    Azerbaijan = "Azerbaijan"
    Bahamas = "Bahamas"
    Bahrain = "Bahrain"
    Bangladesh = "Bangladesh"
    Barbados = "Barbados"
    Belarus = "Belarus"
    Belgium = "Belgium"
    Belize = "Belize"
    Benin = "Benin"
    Bermuda = "Bermuda"
    Bhutan = "Bhutan"
    Bolivia = "Bolivia"
    Bosnia_and_Herzegovina = "Bosnia and Herzegovina"
    Botswana = "Botswana"
    Brazil = "Brazil"
    Brunei = "Brunei"
    Bulgaria = "Bulgaria"
    Burkina_Faso = "Burkina Faso"
    Burundi = "Burundi"
    Cambodia = "Cambodia"
    Cameroon = "Cameroon"
    Canada = "Canada"
    Cape_Verde = "Cape Verde"
    Central_African_Republic = "Central African Republic"
    Chad = "Chad"
    Chile = "Chile"
    China = "China"
    Columbia = "Columbia"
    Comoros = "Comoros"
    Congo = "Congo"
    Costa_Rica = "Costa Rica"
    Croatia = "Croatia"
    Cuba = "Cuba"
    Cyprus = "Cyprus"
    Czech_Republic = "Czech Republic"
    Denmark = "Denmark"
    Djibouti = "Djibouti"
    Dominica = "Dominica"
    East_Timor = "East Timor"
    Ecuador = "Ecuador"
    Egypt = "Egypt"
    El_Salvador = "El Salvador"
    England = "England"
    Equatorial_Guinea = "Equatorial Guinea"
    Eritrea = "Eritrea"
    Estonia = "Estonia"
    Ethiopia = "Ethiopia"
    Fiji = "Fiji"
    Finland = "Finland"
    France = "France"
    Gabon = "Gabon"
    Gambia = "Gambia"
    Georgia = "Georgia"
    Germany = "Germany"
    Ghana = "Ghana"
    Greece = "Greece"
    Grenada = "Grenada"
    Guatemala = "Guatemala"
    Guinea = "Guinea"
    Guinea_Bissau = "Guinea-Bissau"
    Guyana = "Guyana"
    Haiti = "Haiti"
    Honduras = "Honduras"
    Hungary = "Hungary"
    Iceland = "Iceland"
    INDIA = "INDIA"
    Indonesia = "Indonesia"
    Iran = "Iran"
    Iraq = "Iraq"
    Ireland = "Ireland"
    Israel = "Israel"
    Italy = "Italy"
    Ivory_Coast = "Ivory Coast"
    Jamaica = "Jamaica"
    Japan = "Japan"
    Jordan = "Jordan"
    Kazakhstan = "Kazakhstan"
    Kenya = "Kenya"
    Kingdom_of_Saudi_Arabia = "Kingdom of Saudi Arabia"
    Kiribati = "Kiribati"
    Kuwait = "Kuwait"
    Kyrgyz_Republic = "Kyrgyz Republic"
    Laos = "Laos"
    Latvia = "Latvia"
    Lebanon = "Lebanon"
    Lesotho = "Lesotho"
    Liberia = "Liberia"
    Libya = "Libya"
    Liechtenstein = "Liechtenstein"
    Lithuania = "Lithuania"
    Luxembourg = "Luxembourg"
    Macedonia = "Macedonia"
    Madagascar = "Madagascar"
    Malawi = "Malawi"
    Malaysia = "Malaysia"
    Maldives = "Maldives"
    Mali = "Mali"
    Malta = "Malta"
    Marshall_Islands = "Marshall Islands"
    Mauritania = "Mauritania"
    Mauritius = "Mauritius"
    Mexico = "Mexico"
    Micronesia = "Micronesia"
    Moldova = "Moldova"
    Monaco = "Monaco"
    Mongolia = "Mongolia"
    Montenegro = "Montenegro"
    Morocco = "Morocco"
    Mozambique = "Mozambique"
    Myanmar = "Myanmar (formerly known as Burma)"
    Namibia = "Namibia"
    Nauru = "Nauru"
    Nepal = "Nepal"
    Netherlands = "Netherlands"
    New_Zealand = "New Zealand"
    Nicaragua = "Nicaragua"
    Niger = "Niger"
    Nigeria = "Nigeria"
    North_Korea = "North Korea"
    Norway = "Norway"
    Oman = "Oman"
    Others = "Others"
    Pakistan = "Pakistan"
    Palau = "Palau"
    Palestine = "Palestine"
    Panama = "Panama"
    Papua_New_Guinea = "Papua New Guinea"
    Paraguay = "Paraguay"
    Peru = "Peru"
    Philippines = "Philippines"
    Poland = "Poland"
    Portugal = "Portugal"
    Puerto_Rico = "Puerto Rico"
    Qatar = "Qatar"
    Romania = "Romania"
    Russia = "Russia"
    Rwanda = "Rwanda"
    Saint_Kitts_and_Nevis = "Saint Kitts and Nevis"
    Saint_Lucia = "Saint Lucia"
    Samoa = "Samoa"
    San_Marino = "San Marino"
    Sao_Tome_and_Principe = "Sao Tome and Principe"
    Scotland = "Scotland"
    Senegal = "Senegal"
    Serbia = "Serbia"
    Seychelles = "Seychelles"
    Sierra_Leone = "Sierra Leone"
    Singapore = "Singapore"
    Slovakia = "Slovakia"
    Slovenia = "Slovenia"
    Solomon_Islands = "Solomon Islands"
    Somalia = "Somalia"
    South_Africa = "South Africa"
    South_Korea = "South Korea"
    Spain = "Spain"
    Sri_Lanka = "Sri Lanka"
    Sudan = "Sudan"
    Suriname = "Suriname"
    Sweden = "Sweden"
    Switzerland = "Switzerland"
    Syria = "Syria"
    Taiwan = "Taiwan"
    Tajikistan = "Tajikistan"
    Tanzania = "Tanzania"
    Thailand = "Thailand"
    Togo = "Togo"
    Tonga = "Tonga"
    Trinidad_and_Tobago = "Trinidad and Tobago"
    Tunisia = "Tunisia"
    Turkestan = "Turkestan"
    Turkey = "Turkey"
    Turkmenistan = "Turkmenistan"
    Tuvalu = "Tuvalu"
    Uganda = "Uganda"
    Ukraine = "Ukraine"
    United_Arab_Emirates = "United Arab Emirates"
    United_Kingdom = "United Kingdom"
    United_States = "United States"
    Uruguay = "Uruguay"
    Uzbekistan = "Uzbekistan"
    Vanuatu = "Vanuatu"
    Venezuela = "Venezuela"
    Vietnam = "Vietnam"
    Wales = "Wales"
    Yeman = "Yeman"
    Yemen = "Yemen"
    Zambia = "Zambia"
    Zimbabwe = "Zimbabwe"


class Sector(str, Enum):
    GOV_MILITARY = "GOV-MILITARY"
    NONSAUDI_PRIVT = "NONSAUDI_PRIVT"
    GOV_NONMILITARY = "GOV-NONMILITARY"
    PRIVT = "PRIVT"
    NONSAUDI_GOV_NONMILITARY = "NONSAUDI_GOV-NONMILITARY"
    RETMT = "RETMT"
    NULL = "NULL"


class Gender(str, Enum):
    M = "M"
    F = "F"

DATA_PATH = "data"
MODEL_FILENAME = "jarir_model_v3.pkl"

class InputData(BaseModel):
    nationality: str
    age: float
    sector: str
    Gender: str
    simah_score: float
    requested_loan_amount: float
    basic_wage: float
    total_allowance: float
    housing_allowance: float
    svr_jdg_count: float
    svr_jdg_monthsincelastsettld: float
    svr_defaultamount: float
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
            'age', 'simah_score', 'requested_loan_amount',
            'basic_wage', 'total_allowance', 'housing_allowance',
            'svr_jdg_count', 'svr_jdg_monthsincelastsettld',
            'svr_defaultamount', 'days_in_stage'
        ]

    def preprocess_data(self, input_data: InputData):
        data = input_data.dict()

        # Если days_in_stage пришёл — используем его
        # Если нет — рассчитываем по датам
        if data.get('days_in_stage') is not None:
            pass  # уже есть
        else:
            if not data.get("date_of_joining"):
                data["date_of_joining"] = datetime(2023, 1, 1)
            if not data.get("CreatedOn"):
                data["CreatedOn"] = datetime.now()
            date_join = pd.to_datetime(data['date_of_joining'])
            created_on = pd.to_datetime(data['CreatedOn'])
            data['days_in_stage'] = (created_on - date_join).days

        data.pop('date_of_joining', None)
        data.pop('CreatedOn', None)
        return pd.DataFrame([data])


    def predict(self, input_data: InputData, threshold: float = 0.5):
        data_df = self.preprocess_data(input_data)
        cat_indices = self.model.get_cat_feature_indices()
        pool = Pool(data_df, cat_features=[data_df.columns[i] for i in cat_indices])
        proba = self.model.predict_proba(pool)[0, 1]
        decision = "Approve" if proba < threshold else "Reject"
        shap_values_all = self.model.get_feature_importance(pool, type="ShapValues")[0]
        shap_values = shap_values_all[:-1]
        shap_output = [{"feature": f, "value": float(v)} for f, v in zip(data_df.columns, shap_values)]
        return data_df, {
            "probability_of_default": float(proba),
            "decision": decision,
            "shap_values": shap_output
        }