import streamlit as st
import pandas as pd


def render_risk_filter() -> list:
    return st.sidebar.multiselect(
        "Risk Level",
        options=["🔴 HIGH", "🟡 MEDIUM", "🟢 LOW"],
        default=["🔴 HIGH", "🟡 MEDIUM"]
    )


def render_contract_filter(df: pd.DataFrame) -> list:
    return st.sidebar.multiselect(
        "Contract Type",
        options=df["Contract"].unique().tolist(),
        default=df["Contract"].unique().tolist()
    )


def render_probability_filter() -> float:
    return st.sidebar.slider(
        "Min Churn Probability",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05
    )


def render_tenure_filter() -> tuple:
    return st.sidebar.slider(
        "Tenure Range (months)",
        min_value=0,
        max_value=72,
        value=(0, 72)
    )


def apply_filters(
    df: pd.DataFrame,
    risk_filter: list,
    contract_filter: list,
    min_prob: float,
    tenure_range: tuple
) -> pd.DataFrame:
    return df[
        (df["risk_level"].isin(risk_filter)) &
        (df["Contract"].isin(contract_filter)) &
        (df["churn_probability"] >= min_prob) &
        (df["tenure"] >= tenure_range[0]) &
        (df["tenure"] <= tenure_range[1])
    ]