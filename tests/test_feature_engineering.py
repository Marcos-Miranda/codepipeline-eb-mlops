import pandas as pd
from fraud_detector.feature_engineering import (
    create_time_features,
    dataset_split,
    create_text_feature,
    feature_selection,
    set_column_types,
)
import pytest
from math import isclose


@pytest.fixture(scope="function")
def train_df():
    return pd.read_csv("./data/dados_fraude.tsv", sep="\t")


def test_create_time_features(train_df):
    columns = create_time_features(train_df).columns
    assert "fecha" not in columns
    assert "dia_mes" in columns
    assert "dia_semana" in columns
    assert "hora" in columns


def test_dataset_split(train_df):
    train, test = dataset_split(train_df, 0.3)
    total_len = len(train_df)
    train_len = len(train)
    assert isclose(train_len / total_len, 0.7, abs_tol=0.01)
    class1_prop = train_df["fraude"].value_counts(True)[1]
    assert isclose(class1_prop, train["fraude"].value_counts(True)[1], abs_tol=0.01)
    assert isclose(class1_prop, test["fraude"].value_counts(True)[1], abs_tol=0.01)


def test_create_text_feature(train_df):
    train, test = dataset_split(train_df, 0.3)
    train, test = create_text_feature((train, test))
    assert "i" not in train.columns
    assert "i" not in test.columns
    assert "text_feat" in train.columns
    assert "text_feat" in test.columns
    assert train["text_feat"].std() > 0
    assert test["text_feat"].std() > 0


def test_feature_selection(train_df):
    train_df = create_time_features(train_df)
    train, test = dataset_split(train_df)
    train, test = create_text_feature((train, test))
    n_feats = len(train.columns) - 2
    train, test = feature_selection((train, test))
    n_selected_feats = len(train.columns) - 2
    assert n_selected_feats / n_feats > 0.5


def test_set_column_types(train_df):
    train_df = create_time_features(train_df)
    train, test = dataset_split(train_df)
    train, test = create_text_feature((train, test))
    train, test = set_column_types((train, test))
    assert train["n"].dtype.name == "category"
    train.drop(columns=["id", "fraude"], inplace=True)
    assert "O" not in train.dtypes.to_list()
