import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from boruta import BorutaPy
from typing import Tuple
import pickle
import json

CAT_FEATURES = ["g", "j", "n", "o", "p", "dia_semana"]
NUM_FEATURES = ["a", "b", "c", "d", "e", "f", "h", "k", "l", "m", "monto", "dia_mes", "hora"]


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-related features from the transaction's timestamp."""
    time_column = "fecha"
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
    df["dia_mes"] = df[time_column].dt.day
    df["dia_semana"] = df[time_column].dt.day_name()
    df["hora"] = df[time_column].dt.round("H").dt.hour
    df.drop(columns=time_column, inplace=True)
    return df


def dataset_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into train and test sets."""
    train_set, test_set = train_test_split(df, test_size=test_size, stratify=df["fraude"], random_state=15)
    return (train_set, test_set)


def create_text_feature(dfs: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a feature from the text column by training a Logistic Regression with Bag of Words."""
    pipe = Pipeline(
        [
            ("bow", CountVectorizer(ngram_range=(1, 2))),
            ("over", RandomOverSampler(random_state=15)),
            ("lr", LogisticRegression(max_iter=1000, n_jobs=-1)),
        ]
    )
    text_column = "i"
    train_set, test_set = dfs
    train_set["text_feat"] = cross_val_predict(
        pipe, train_set[text_column], train_set["fraude"], cv=10, method="predict_proba"
    )[:, 1]
    pipe.fit(train_set[text_column], train_set["fraude"])
    pickle.dump(pipe, open("../models/text_model.pkl", "wb"))
    test_set["text_feat"] = pipe.predict_proba(test_set[text_column])[:, 1]
    train_set.drop(columns=text_column, inplace=True)
    test_set.drop(columns=text_column, inplace=True)
    return (train_set, test_set)


def feature_selection(dfs: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select the most import features with Boruta."""
    train_set, test_set = dfs
    X_boruta = train_set.drop(columns=["fraude", "id"])
    for col in CAT_FEATURES:
        X_boruta[col] = LabelEncoder().fit_transform(X_boruta[col].fillna("missing"))
    for col in NUM_FEATURES:
        X_boruta[col] = X_boruta[col].fillna(X_boruta[col].mean())
    rf_model = RandomForestClassifier(max_depth=5, class_weight="balanced", n_jobs=-1)
    boruta = BorutaPy(rf_model, n_estimators="auto", verbose=2)
    boruta.fit(X_boruta.values, train_set["fraude"].values)
    selected_features = X_boruta.columns[boruta.support_].tolist()
    json.dump({"columns": selected_features}, open("../models/selected_features.json", "w"))
    train_set = train_set[selected_features + ["id", "fraude"]]
    test_set = test_set[selected_features + ["id", "fraude"]]
    return (train_set, test_set)


def set_column_types(dfs: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Set the type of categorical features to 'category' for LightGBM."""
    train_set, test_set = dfs
    if "n" in train_set.columns:
        train_set["n"] = np.select(
            [train_set["n"] == 1, train_set["n"] == 0, train_set["n"].isna()], ["Y", "N", np.nan]
        )
        test_set["n"] = np.select([test_set["n"] == 1, test_set["n"] == 0, test_set["n"].isna()], ["Y", "N", np.nan])
    for col in set(CAT_FEATURES).intersection(set(train_set.columns)):
        train_set[col] = train_set[col].fillna("missing")
        train_set[col] = train_set[col].astype("category")
        test_set[col] = test_set[col].fillna("missing")
        test_set[col] = test_set[col].astype("category")
    return (train_set, test_set)


def feat_engineering_pipe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline for feature engineering."""
    df = create_time_features(df)
    train_set, test_set = dataset_split(df)
    train_set, test_set = create_text_feature((train_set, test_set))
    train_set, test_set = feature_selection((train_set, test_set))
    train_set, test_set = set_column_types((train_set, test_set))
    json.dump({"cat_features": CAT_FEATURES}, open("../models/categorical_features.json", "w"))
    return (train_set, test_set)
