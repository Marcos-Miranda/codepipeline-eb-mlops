from fraud_detector.prediction import predict
import pytest


@pytest.fixture(scope="function")
def full_input():
    ex_input = {
        "id": "ea767f33-fdc6-4722-b0bc-1b393da86870",
        "a": 4,
        "b": 0.7478,
        "c": 40073.59,
        "d": 17.0,
        "e": 0.0,
        "f": 5.0,
        "g": "BR",
        "h": 34,
        "i": "Kit 30 Esmaltes Anita*você Escolhe As Cores* Nova Coleção",
        "j": "cat_1df0a5f",
        "k": 0.4905986270729947,
        "l": 991.0,
        "m": 88.0,
        "n": 1,
        "o": "Y",
        "p": "N",
        "fecha": "2020-03-13 15:33:48",
        "monto": 34.39,
    }
    return ex_input


def test_full_input(full_input):
    assert "fraude" in predict(full_input)


def test_null_values(full_input):
    keys = list(full_input.keys())[1:]
    for key in keys:
        del full_input[key]
        assert "fraude" in predict(full_input)


def test_new_cat_values(full_input):
    full_input["g"] = "new"
    assert "fraude" in predict(full_input)
    full_input["j"] = "new"
    assert "fraude" in predict(full_input)
    full_input["o"] = "new"
    assert "fraude" in predict(full_input)
    full_input["p"] = "new"
    assert "fraude" in predict(full_input)
