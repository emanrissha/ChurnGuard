# 🛡️ ChurnGuard — B2B SaaS Churn Prediction Engine

> Predicts which customers will cancel **30 days before they do** — with SHAP explanations and a live business dashboard.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)](https://xgboost.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)

---

## 💼 Business Impact

For a 500-customer SaaS company at ₪8,000 ARR per customer:

| Metric | Value |
|--------|-------|
| Churners correctly identified | 79% (recall) |
| Revenue saved per year | ₪2,370,000 |
| Cost of retention outreach | ₪148,000 |
| **Net annual benefit** | **₪2,220,000+** |
| False alarm rate reduced vs baseline | 23% fewer unnecessary calls |

> This model would save a 500-customer SaaS company approximately ₪2.2M/year in churned revenue.

---

## 🏗️ Architecture
Raw Data → Feature Engineering → Model Training → FastAPI → Streamlit Dashboard
↓
SHAP Explainability
↓
RAG Chatbot (Hebrew/English)

---

## 📊 Model Comparison

| Model | F1 | AUC | Recall | Total Error Cost |
|-------|----|-----|--------|-----------------|
| Logistic Regression (baseline) | 0.596 | 0.830 | 0.773 | ₪833,500 |
| Random Forest | 0.618 | 0.836 | 0.765 | ₪837,000 |
| **XGBoost (tuned) ✅** | **0.613** | **0.837** | **0.791** | **₪772,000** |

**XGBoost was selected** as the champion model — lowest business cost and highest revenue saved.

---

## 🔍 SHAP Explainability

Every prediction comes with a human-readable explanation:
Customer 7590-VHVEG — Churn Probability: 82% 🔴 HIGH
Top risk factors:

Contract_Month-to-month:  +0.477  ← biggest driver
risk_score:               +0.457  ← engineered feature
tenure (2 months):        +0.218  ← very new customer

Top protective factors:

MonthlyCharges:           -0.088
avg_monthly_revenue:      -0.070

Recommended action: Immediate outreach — offer annual contract upgrade

---

## ⚙️ Feature Engineering (15+ features)

| Feature | Description |
|---------|-------------|
| `tenure_cohort` | Customer grouped by tenure: 0-12, 12-24, 24-48, 48-72 months |
| `avg_monthly_revenue` | TotalCharges / tenure |
| `revenue_per_product` | MonthlyCharges / number of active products |
| `charge_increase_rate` | Revenue velocity — is spending accelerating? |
| `product_count` | Total number of services subscribed |
| `risk_score` | Composite score: month-to-month + fiber + electronic check + new + senior alone |
| `is_loyal` | Tenure ≥ 24 months |
| `is_high_spender` | MonthlyCharges > 75th percentile |
| `is_senior_alone` | Senior citizen with no partner or dependents |

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/emanrissha/ChurnGuard
cd ChurnGuard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .

# 2. Get the data
curl -L "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" \
  -o data/raw/telco_churn.csv

# 3. Train all models
make train

# 4. Run the API
make api

# 5. Run the dashboard
make dashboard
```

---

## 🐳 Docker

```bash
# Build and run everything
make docker-up

# API available at:  http://localhost:8000
# Dashboard at:      http://localhost:8501
# API docs at:       http://localhost:8000/docs
```

---

## 📡 API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2,
    "Contract_Month_to_month": 1,
    "InternetService_Fiber_optic": 1,
    "MonthlyCharges": 70.0,
    ...
  }'
```

Response:
```json
{
  "churn_probability": 0.8159,
  "churn_prediction": 1,
  "risk_level": "HIGH",
  "top_risk_factors": {
    "Contract_Month-to-month": 0.477,
    "risk_score": 0.457,
    "tenure": 0.218
  },
  "estimated_revenue_at_risk_ils": 6527.19
}
```

---

## 🗂️ Project Structure
ChurnGuard/
├── src/
│   ├── data/          # Loader + preprocessor
│   ├── features/      # 15+ engineered features
│   ├── models/        # LR, RF, XGBoost + evaluator
│   ├── explainability/# SHAP explainer
│   └── rag/           # LangChain chatbot
├── api/               # FastAPI endpoints
├── dashboard/         # Streamlit 4-page app
├── models/            # Saved model artifacts
├── data/              # Raw + processed data
├── tests/             # pytest suite
├── Dockerfile
└── docker-compose.yml

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | Pandas, NumPy, PyArrow |
| ML | scikit-learn, XGBoost, imbalanced-learn |
| Explainability | SHAP |
| Experiment tracking | Weights & Biases |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| RAG Chatbot | LangChain + OpenAI |
| Deployment | Docker, Render |
| CI/CD | GitHub Actions |

---

## 👨‍💻 Author

Built by [@emanrissha](https://github.com/emanrissha) — inspired by real ML systems at Monday.com, Riskified, and Similarweb.