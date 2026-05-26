import os

def test_svm_model_exists():
    assert os.path.exists("models/svm_model.pkl")

def test_rf_model_exists():
    assert os.path.exists("models/rf_model.pkl")