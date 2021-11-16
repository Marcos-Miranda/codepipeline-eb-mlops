import pandas as pd
from src.feature_engineering import feat_engineering_pipe
from src.model_training import train_model
import pytest
import os
from pathlib import Path


@pytest.fixture(scope="function")
def train_df():
    return pd.read_csv(Path(__file__).parent / "dados_fraude.tsv", sep="\t")


@pytest.fixture(scope="function")
def change_test_dir(request):
    os.chdir(request.fspath.dirname)
    yield
    os.chdir(request.config.invocation_dir)


def test_train_model(train_df, change_test_dir):
    train, test = feat_engineering_pipe(train_df)
    model, test_metrics = train_model((train, test))
    assert test_metrics.loc["class_0", "recall"] > 0.5
    assert test_metrics.loc["class_1", "recall"] > 0.5
    model.predict(test.drop(columns=["id", "fraude"]).iloc[0:1])
