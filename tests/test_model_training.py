import pandas as pd
from fraud_detector.feature_engineering import feat_engineering_pipe
from fraud_detector.model_training import train_model
import pytest


@pytest.fixture(scope="function")
def train_df():
    return pd.read_csv("./data/dados_fraude.tsv", sep="\t")


def test_train_model(train_df):
    train, test = feat_engineering_pipe(train_df)
    model, test_metrics = train_model((train, test))
    assert test_metrics.loc["class_0", "recall"] > 0.5
    assert test_metrics.loc["class_1", "recall"] > 0.5
    model.predict(test.drop(columns=["id", "fraude"]).iloc[0:1])
