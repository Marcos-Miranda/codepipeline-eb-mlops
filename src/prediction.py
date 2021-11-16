import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd

SELECTED_FEATURES = json.load(open("../models/selected_features.json"))
CATEGORICAL_FEATURES = json.load(open("../models/categorical_features.json"))
TEXT_MODEL = pickle.load(open("../models/text_model.pkl", "rb"))
MODEL = pickle.load(open("../models/model.pkl", "rb"))


def predict(data: dict) -> dict:
    """Transform the input data and run the model's prediction on it."""
    if "fecha" in data and data["fecha"] is not None:
        dt = datetime.strptime(data["fecha"], "%Y-%m-%d %H:%M:%S")
        data["dia_mes"] = dt.day
        data["dia_semana"] = dt.strftime("%A")
        data["hora"] = dt.hour if dt.minute < 30 else dt.hour + 1
    if "i" in data and data["i"] is not None:
        data["text_feat"] = TEXT_MODEL.predict_proba([data["i"]])[0][1]
    if "n" in data and data["n"] is not None:
        data["n"] = "Y" if data["n"] == 1 else "N"
    model_input = []
    for column in SELECTED_FEATURES["columns"]:
        if column in data and data[column] is not None:
            model_input.append(data[column])
        else:
            if column in CATEGORICAL_FEATURES:
                model_input.append("missing")
            else:
                model_input.append(np.nan)
    x = pd.DataFrame([model_input], columns=SELECTED_FEATURES["columns"])
    for col in x.columns:
        if col in CATEGORICAL_FEATURES:
            x[col] = x[col].astype("category")
    pred = MODEL.predict_proba(x)[0][1]
    return {"fraude": "sim" if pred > 0.5 else "nao", "probabilidade_fraude": pred}
