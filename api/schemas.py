from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: int = Field(..., ge=0, le=1)
    Dependents: int = Field(..., ge=0, le=1)
    tenure: int = Field(..., ge=0, le=72)
    PhoneService: int = Field(..., ge=0, le=1)
    MultipleLines: int = Field(..., ge=0, le=1)
    OnlineSecurity: int = Field(..., ge=0, le=1)
    OnlineBackup: int = Field(..., ge=0, le=1)
    DeviceProtection: int = Field(..., ge=0, le=1)
    TechSupport: int = Field(..., ge=0, le=1)
    StreamingTV: int = Field(..., ge=0, le=1)
    StreamingMovies: int = Field(..., ge=0, le=1)
    PaperlessBilling: int = Field(..., ge=0, le=1)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    InternetService_DSL: int = Field(..., ge=0, le=1)
    InternetService_Fiber_optic: int = Field(..., ge=0, le=1)
    InternetService_No: int = Field(..., ge=0, le=1)
    Contract_Month_to_month: int = Field(..., ge=0, le=1)
    Contract_One_year: int = Field(..., ge=0, le=1)
    Contract_Two_year: int = Field(..., ge=0, le=1)
    PaymentMethod_Bank_transfer: int = Field(..., ge=0, le=1)
    PaymentMethod_Credit_card: int = Field(..., ge=0, le=1)
    PaymentMethod_Electronic_check: int = Field(..., ge=0, le=1)
    PaymentMethod_Mailed_check: int = Field(..., ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    top_risk_factors: dict
    top_protective_factors: dict
    estimated_revenue_at_risk_ils: float


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    version: str