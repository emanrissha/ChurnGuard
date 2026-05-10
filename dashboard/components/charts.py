import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_risk_distribution(df: pd.DataFrame) -> go.Figure:
    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Customers"]
    fig = px.pie(
        risk_counts,
        values="Customers",
        names="Risk Level",
        color_discrete_sequence=["#e74c3c", "#f39c12", "#2ecc71"],
        title="Customer Risk Distribution"
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    return fig


def plot_churn_by_contract(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="Contract",
        color="risk_level",
        barmode="group",
        title="Churn Risk by Contract Type",
        color_discrete_map={
            "🔴 HIGH": "#e74c3c",
            "🟡 MEDIUM": "#f39c12",
            "🟢 LOW": "#2ecc71"
        }
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    return fig


def plot_revenue_at_risk(df: pd.DataFrame) -> go.Figure:
    contract_risk = df.groupby("Contract")["revenue_at_risk_ils"].sum().reset_index()
    fig = px.bar(
        contract_risk,
        x="Contract",
        y="revenue_at_risk_ils",
        title="Revenue at Risk by Contract Type (₪)",
        color="Contract",
        color_discrete_sequence=["#e74c3c", "#f39c12", "#2ecc71"]
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    return fig


def plot_probability_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="churn_probability",
        nbins=30,
        title="Churn Probability Distribution",
        color_discrete_sequence=["#3498db"]
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    return fig


def plot_tenure_vs_churn(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="tenure",
        y="churn_probability",
        color="risk_level",
        title="Tenure vs Churn Probability",
        color_discrete_map={
            "🔴 HIGH": "#e74c3c",
            "🟡 MEDIUM": "#f39c12",
            "🟢 LOW": "#2ecc71"
        },
        opacity=0.6
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    return fig