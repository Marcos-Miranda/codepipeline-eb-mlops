import pandas as pd
from fraud_detector.model_training import model_training_pipe
import pytest


@pytest.fixture(scope="function")
def train_df():
    return pd.read_csv("./data/dados_fraude.tsv", sep="\t")


def test_train_model(train_df):
    test_metrics = model_training_pipe(train_df)
    assert test_metrics.loc["class_0", "recall"] > 0.5
    assert test_metrics.loc["class_1", "recall"] > 0.5
