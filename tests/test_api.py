import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 0,
    "Dependents": 0,
    "tenure": 2,
    "PhoneService": 1,
    "MultipleLines": 0,
    "OnlineSecurity": 0,
    "OnlineBackup": 0,
    "DeviceProtection": 0,
    "TechSupport": 0,
    "StreamingTV": 0,
    "StreamingMovies": 0,
    "PaperlessBilling": 1,
    "MonthlyCharges": 70.0,
    "TotalCharges": 140.0,
    "InternetService_DSL": 0,
    "InternetService_Fiber_optic": 1,
    "InternetService_No": 0,
    "Contract_Month_to_month": 1,
    "Contract_One_year": 0,
    "Contract_Two_year": 0,
    "PaymentMethod_Bank_transfer": 0,
    "PaymentMethod_Credit_card": 0,
    "PaymentMethod_Electronic_check": 1,
    "PaymentMethod_Mailed_check": 0,
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["project"] == "ChurnGuard"


def test_predict_returns_200():
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200


def test_predict_response_structure():
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data
    assert "top_risk_factors" in data
    assert "estimated_revenue_at_risk_ils" in data


def test_predict_probability_range():
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_high_risk_customer():
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    data = response.json()
    assert data["risk_level"] == "HIGH"
    assert data["churn_probability"] >= 0.7


def test_predict_invalid_input():
    response = client.post("/predict", json={"tenure": "invalid"})
    assert response.status_code == 422