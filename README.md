# 🛡️ ChurnGuard — AI Churn Prediction Engine

[![CI](https://github.com/emanrissha/ChurnGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/emanrissha/ChurnGuard/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)](https://xgboost.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)


> Predicts which customers will cancel **30 days before they do** — with SHAP explanations and a live business dashboard.

> **Live Dashboard:** https://churnguard-dashboard.onrender.com

> **Live API:** https://churnguard-api-9kut.onrender.com/docs

---

## 📖 The Story

Imagine you run a B2B SaaS company with 500 customers.

Every month, some of them cancel. You find out when they click "Cancel Subscription." By then, it's too late — the decision was made weeks ago, when they stopped logging in, when their support tickets went unanswered, when a competitor caught their eye.

**The problem isn't churn. The problem is that you don't see it coming.**

Customer Success teams are reactive. They call customers after they've already decided to leave. Sales teams chase new customers while ignoring the ones slipping away. And every churned customer represents thousands of shekels in lost annual recurring revenue — gone forever.

This is the reality for most SaaS companies in Israel and worldwide.

---

## 💡 The Solution

**ChurnGuard** is an end-to-end AI system that predicts which customers will churn — **30 days before they cancel.**

It doesn't just flag a customer as "at risk." It tells you **why** they're at risk, **how much revenue** you stand to lose, and **what your CS team should do right now** to save them.

Built the way Monday.com, Riskified, and Similarweb actually build ML systems — not a Jupyter notebook, a production system.

---

## 🎯 What We Solve

| Problem | ChurnGuard Solution |
|---------|-------------------|
| "We only know customers churned after they cancel" | Predicts churn 30 days in advance with 83.7% AUC |
| "We don't know which customers to prioritize" | Ranks all customers by churn probability and ₪ revenue at risk |
| "We know WHO will churn but not WHY" | SHAP explainability — exact reasons per customer |
| "Our CS team doesn't know what action to take" | Claude AI chatbot gives actionable recommendations in Hebrew or English |
| "We can't justify the budget for retention efforts" | Business cost calculator shows exact ROI per intervention |

---

## 📊 The Results

Trained and evaluated on **7,032 real B2B SaaS customers:**

| Metric | Value |
|--------|-------|
| Model | XGBoost (tuned) |
| AUC Score | **0.837** |
| Recall | **79.1%** — catches 4 out of 5 churners |
| Precision | 50.0% |
| F1 Score | 0.613 |

### Business Impact (500-customer SaaS company at ₪8,000 ARR)

| Metric | Value |
|--------|-------|
| High-risk customers identified | ~185 |
| Revenue at risk | ₪1,480,000 |
| Cost of retention outreach | ₪92,500 |
| Expected customers saved (40% success rate) | ~74 |
| **Annual revenue saved** | **₪592,000** |
| **Net annual benefit** | **₪499,500** |
| **ROI multiple** | **6.4x** |

> For a larger company at 2,000 customers: **~₪2.37M saved annually.**

---

## 🏗️ How It Works
Raw Data (7,043 customers, 21 features)

↓

Data Cleaning
(fix TotalCharges, encode categoricals)
↓
Feature Engineering
(13 new features: risk_score, tenure_cohort,
avg_monthly_revenue, product_count...)
↓
Model Training & Selection
Logistic Regression → baseline
Random Forest       → ensemble
XGBoost (tuned)     → champion ✅
↓
SHAP Explainability
(global importance + per-customer waterfall)
↓
FastAPI
/predict → churn probability + SHAP
/explain → full customer breakdown
/health  → service status
↓
Streamlit Dashboard
Risk Table | Deep Dive | Business Impact | Chatbot
↓
Claude AI Chatbot
"למה הלקוח הזה בסיכון?" → Hebrew answer
"Why is this customer at risk?" → English answer

---

## 🔍 SHAP Explainability — The "Why" Behind Every Prediction

Most ML models are black boxes. ChurnGuard explains every prediction in plain language.

**Example — Customer 7590-VHVEG (83.3% churn probability):**

🔴 HIGH RISK — Churn Probability: 83.3%
Top risk factors:
↑ Month-to-month contract    +0.459  (biggest driver)
↑ Risk score (composite)     +0.457  (engineered feature)
↑ Only 1 month tenure        +0.386  (new customer)
↑ High total charges         +0.116
↑ No long-term commitment    +0.087
Top protective factors:
↓ Low monthly charges        -0.112  (price not the issue)
↓ Avg monthly revenue        -0.087
Recommended action: Call within 48 hours.
Offer 20% discount on annual contract upgrade.
Revenue at risk: ₪6,666

---

---

## 🤖 Claude AI Chatbot — Ask in Hebrew or English

The most unique feature of ChurnGuard. Business users — not data scientists — can ask questions about any customer in natural language.

**Example conversation:**

> **CS Manager:** למה הלקוח 7590-VHVEG בסיכון?
>
> **ChurnGuard AI:** הלקוח בסיכון גבוה מאוד (83.3%) בגלל שלושה גורמים מרכזיים: הוא לקוח חדש מאוד (חודש אחד בלבד), עם חוזה חודשי ללא התחייבות, ותשלום בצ'ק אלקטרוני. המלצה: התקשרו ללקוח תוך 48 שעות עם הצעה לחוזה שנתי עם 20% הנחה.

> **CS Manager:** What should I do immediately?
>
> **ChurnGuard AI:** Initiate a proactive welcome call within 48 hours focused on onboarding success, then present an annual contract offer with a time-limited incentive before day 14. The goal: convert from month-to-month before they mentally "trial and leave."

---

## 🧠 Feature Engineering — 13 Custom Features

Raw data has 21 columns. ChurnGuard engineers 13 additional features that capture behavioral signals:

| Feature | Description | Business Logic |
|---------|-------------|----------------|
| `risk_score` | Composite danger score (0-6) | Month-to-month + fiber + electronic check + new + senior alone |
| `tenure_cohort` | Loyalty group (1-4) | 0-12m, 12-24m, 24-48m, 48-72m |
| `avg_monthly_revenue` | TotalCharges / tenure | Spending trend over time |
| `revenue_per_product` | MonthlyCharges / active products | Value per service |
| `charge_increase_rate` | Revenue velocity | Is spending accelerating? |
| `product_count` | Number of active services | Depth of product adoption |
| `is_loyal` | Tenure ≥ 24 months | Long-term relationship flag |
| `is_high_spender` | Charges > 75th percentile | Premium customer flag |
| `is_senior_alone` | Senior, no partner, no dependents | Vulnerable segment |
| `is_month_to_month` | Contract type flag | Biggest churn signal |
| `is_electronic_check` | Payment method flag | Low commitment indicator |
| `is_fiber` | Fiber internet flag | High-cost, high-churn segment |

---

## 📊 Model Comparison

| Model | F1 | AUC | Recall | Error Cost | Saved Revenue |
|-------|----|-----|--------|------------|---------------|
| Logistic Regression | 0.596 | 0.830 | 0.773 | ₪833,500 | ₪2,312,000 |
| Random Forest | 0.618 | 0.836 | 0.765 | ₪837,000 | ₪2,288,000 |
| **XGBoost ✅** | **0.613** | **0.837** | **0.791** | **₪772,000** | **₪2,368,000** |

**XGBoost selected** — lowest business cost and highest revenue saved.

---

## 🌐 Live Deployment

| Service | URL |
|---------|-----|
| 📊 Dashboard | https://churnguard-dashboard.onrender.com |
| 🔌 API Health | https://churnguard-api-9kut.onrender.com/health |
| 📖 API Docs | https://churnguard-api-9kut.onrender.com/docs |
| 🔮 Explain Endpoint | https://churnguard-api-9kut.onrender.com/explain/7590-VHVEG |

---

## 📡 API Usage

### Predict churn probability:
```bash
curl -X POST https://churnguard-api-9kut.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2,
    "Contract_Month_to_month": 1,
    "InternetService_Fiber_optic": 1,
    "MonthlyCharges": 70.0,
    "TotalCharges": 140.0,
    ...
  }'
```

### Response:
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

### Explain a specific customer:
```bash
curl https://churnguard-api-9kut.onrender.com/explain/7590-VHVEG
```

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/emanrissha/ChurnGuard
cd ChurnGuard

# 2. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .

# 3. Train
make train

# 4. Run API
make api
# → http://localhost:8000/docs

# 5. Run Dashboard
make dashboard
# → http://localhost:8501
```

---

## 🐳 Docker

```bash
# Run everything with one command
make docker-up

# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## 🗂️ Project Structure


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
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| AI Chatbot | Claude AI (Anthropic) |
| Testing | pytest + coverage |
| CI/CD | GitHub Actions |
| Deployment | Docker + Render |

---

## ✅ Test Suite

```bash
pytest tests/ -v
# 23 passed in 7.77s
```

| Test File | Coverage |
|-----------|----------|
| `test_features.py` | Data cleaning, feature engineering, encoding |
| `test_model.py` | Model loading, predictions, AUC/recall thresholds |
| `test_api.py` | All endpoints, response structure, edge cases |

---

## 👨‍💻 Author

Built by [@emanrissha](https://github.com/emanrissha)

Inspired by real ML systems at Monday.com, Riskified, and Similarweb.
Built to show what a production ML project looks like — not a tutorial, a system.

---
