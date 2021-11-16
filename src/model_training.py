import numpy as np
import pandas as pd
from feature_engineering import feat_engineering_pipe
import optuna
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split
from functools import partial
from typing import Tuple, Union
import pickle


def get_model() -> Pipeline:
    """Return a instance of the model's pipeline."""
    lgbm_pipe = Pipeline(
        [
            ("over", RandomOverSampler()),
            ("lgbm", LGBMClassifier(n_jobs=-1)),
        ]
    )
    return lgbm_pipe


def obj(
    trial: optuna.trial.Trial,
    train_data: Tuple[pd.DataFrame, pd.DataFrame],
    eval_data: Tuple[pd.DataFrame, pd.DataFrame],
) -> float:
    """Objective function of the hyperparameter optimization."""
    params = {
        "over__sampling_strategy": trial.suggest_float("over__sampling_strategy", 0.5, 1.0),
        "lgbm__learning_rate": trial.suggest_float("lgbm__learning_rate", 1e-4, 5e-1, log=True),
        "lgbm__reg_alpha": trial.suggest_float("lgbm__reg_alpha", 1e-3, 1.0, log=True),
        "lgbm__reg_lambda": trial.suggest_float("lgbm__reg_lambda", 1e-3, 1.0, log=True),
        "lgbm__subsample": trial.suggest_float("lgbm__subsample", 0.4, 1.0),
        "lgbm__colsample_bytree": trial.suggest_float("lgbm__colsample_bytree", 0.4, 1.0),
        "lgbm__min_child_samples": trial.suggest_int("lgbm__min_child_samples", 1, 100, 1),
        "lgbm__num_leaves": trial.suggest_int("lgbm__num_leaves", 2, 50, 1),
        "lgbm__subsample_freq": trial.suggest_int("lgbm__subsample_freq", 1, 10, 1),
        "lgbm__n_estimators": trial.suggest_int("lgbm__n_estimators", 100, 5000, 1),
    }

    model = get_model().set_params(**params)
    model.fit(train_data[0], train_data[1])
    preds = model.predict(eval_data[0])
    return recall_score(eval_data[1], preds)


def get_test_metrics(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
    """Compute performance metrics on the test set."""
    recall1 = recall_score(y_true, y_pred)
    recall0 = recall_score(y_true, y_pred, pos_label=0)
    precision1 = precision_score(y_true, y_pred)
    precision0 = precision_score(y_true, y_pred, pos_label=0)
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true)
    proportions = y_true.value_counts(normalize=True)
    metrics = pd.DataFrame(
        data={
            "recall": [recall0, recall1],
            "precision": [precision0, precision1],
            "proportion": [proportions[0], proportions[1]],
        },
        index=["class_0", "class_1"],
    )
    return metrics


def train_model(dfs: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[Pipeline, pd.DataFrame]:
    """Train and validate the model."""
    train_set, test_set = dfs
    X = train_set.drop(columns=["id", "fraude"])
    y = train_set["fraude"]
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, stratify=y, test_size=0.1, random_state=15)
    study = optuna.create_study(direction="maximize")
    study.optimize(partial(obj, train_data=(X_train, y_train), eval_data=(X_eval, y_eval)), n_trials=50)
    model = get_model().set_params(**study.best_params)
    model.fit(X, y)
    test_preds = model.predict(test_set.drop(columns=["id", "fraude"]))
    test_metrics = get_test_metrics(test_set["fraude"], test_preds)
    return (model, test_metrics)


def model_training_pipe(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the entire pipeline of model training, including the steps of feature engineering
    and selection and hyperparameter tuning. The resulting model is saved in the file "model.pkl".

    Args:
        df (pandas DataFrame): Dataset to train and test the model.

    Returns:
        pandas DataFrame: metrics on the test set.
    """
    train_data, test_data = feat_engineering_pipe(df)
    model, test_metrics = train_model((train_data, test_data))
    pickle.dump(model, open("../models/model.pkl", "wb"))
    return test_metrics
