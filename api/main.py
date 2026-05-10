from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.middleware import log_requests
from api.routers import predict, health, explain

app = FastAPI(
    title="ChurnGuard API",
    description="Predicts B2B SaaS customer churn probability with SHAP explanations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(log_requests)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)


@app.get("/", tags=["Root"])
def root():
    return {
        "project": "ChurnGuard",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }