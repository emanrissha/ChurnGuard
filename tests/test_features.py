import pytest
import pandas as pd
import numpy as np
from src.data.loader import load_raw_data
from src.data.preprocessor import clean_data, get_features_and_target
from src.features.engineering import engineer_features


@pytest.fixture
def clean_df():
    return clean_data(load_raw_data())


@pytest.fixture
def engineered_df(clean_df):
    return engineer_features(clean_df)


def test_clean_data_shape(clean_df):
    assert clean_df.shape[0] > 7000
    assert clean_df.shape[1] == 27


def test_no_missing_values(clean_df):
    assert clean_df.isnull().sum().sum() == 0


def test_churn_is_binary(clean_df):
    assert set(clean_df["Churn"].unique()) == {0, 1}


def test_total_charges_is_numeric(clean_df):
    assert clean_df["TotalCharges"].dtype in [np.float64, np.float32]


def test_engineer_features_adds_columns(clean_df, engineered_df):
    assert engineered_df.shape[1] > clean_df.shape[1]


def test_risk_score_range(engineered_df):
    assert engineered_df["risk_score"].min() >= 0
    assert engineered_df["risk_score"].max() <= 6


def test_product_count_range(engineered_df):
    assert engineered_df["product_count"].min() >= 0
    assert engineered_df["product_count"].max() <= 7


def test_tenure_cohort_values(engineered_df):
    assert set(engineered_df["tenure_cohort"].unique()).issubset({1, 2, 3, 4})


def test_binary_flags_are_binary(engineered_df):
    binary_cols = ["is_loyal", "is_high_spender", "is_senior_alone",
                   "is_month_to_month", "is_electronic_check", "is_fiber"]
    for col in binary_cols:
        assert set(engineered_df[col].unique()).issubset({0, 1}), f"{col} is not binary"


def test_feature_target_split(engineered_df):
    X, y = get_features_and_target(engineered_df)
    assert "Churn" not in X.columns
    assert len(X) == len(y)
    assert set(y.unique()) == {0, 1}