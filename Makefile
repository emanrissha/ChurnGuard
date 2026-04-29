.PHONY: install train api dashboard test lint docker-build docker-up

install:
	pip install -r requirements-dev.txt
	pip install -e .

train:
	python -m src.data.preprocessor
	python -m src.features.engineering
	python -m src.models.xgboost_model
	python -m src.explainability.shap_explainer

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py --server.port 8501

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ api/ dashboard/
	black --check src/ api/ dashboard/

docker-build:
	docker build -t churnguard .

docker-up:
	docker-compose up --build

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete