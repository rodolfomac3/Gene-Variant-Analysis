# Makefile for Gene Variant Analysis Pipeline

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy

# Directories
SRC_DIR := src
TESTS_DIR := tests
SCRIPTS_DIR := scripts
CONFIG_DIR := config
DATA_DIR := data
MODELS_DIR := models

# Default target
.DEFAULT_GOAL := help

# Help target
help: ## Show this help message
	@echo "Gene Variant Analysis Pipeline - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Setup targets
install: ## Install dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

setup: install-dev ## Setup development environment
	pre-commit install
	$(PYTHON) -m pip install -e .

# Data targets
create-dirs: ## Create necessary directories
	mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed $(DATA_DIR)/external $(DATA_DIR)/interim
	mkdir -p $(MODELS_DIR)/trained $(MODELS_DIR)/artifacts
	mkdir -p logs mlruns

download-data: ## Download sample data (placeholder)
	@echo "Downloading sample data..."
	@echo "This would download genomic data from public repositories"

prepare-data: ## Prepare and validate data
	$(PYTHON) scripts/prepare_data.py --vcf $(DATA_DIR)/raw/variants.vcf --output $(DATA_DIR)/processed/

# Training targets
train: ## Train the model
	$(PYTHON) scripts/train.py --vcf $(DATA_DIR)/raw/variants.vcf --model-type xgboost

train-ensemble: ## Train ensemble model
	$(PYTHON) scripts/train.py --vcf $(DATA_DIR)/raw/variants.vcf --ensemble

train-all: ## Train all model types
	$(PYTHON) scripts/train.py --vcf $(DATA_DIR)/raw/variants.vcf --model-type xgboost
	$(PYTHON) scripts/train.py --vcf $(DATA_DIR)/raw/variants.vcf --model-type lightgbm
	$(PYTHON) scripts/train.py --vcf $(DATA_DIR)/raw/variants.vcf --model-type random_forest

# Evaluation targets
evaluate: ## Evaluate trained model
	$(PYTHON) scripts/evaluate.py --model $(MODELS_DIR)/trained/model.pkl --test-data $(DATA_DIR)/processed/test.csv

predict: ## Make predictions
	$(PYTHON) scripts/predict.py --model $(MODELS_DIR)/trained/model.pkl --data $(DATA_DIR)/raw/new_variants.vcf

# Code quality targets
format: ## Format code with black and isort
	$(BLACK) $(SRC_DIR) $(SCRIPTS_DIR) $(TESTS_DIR)
	$(ISORT) $(SRC_DIR) $(SCRIPTS_DIR) $(TESTS_DIR)

lint: ## Run linting
	$(FLAKE8) $(SRC_DIR) $(SCRIPTS_DIR) --max-line-length=88 --extend-ignore=E203,W503
	$(MYPY) $(SRC_DIR) --ignore-missing-imports

check-format: ## Check code formatting
	$(BLACK) --check $(SRC_DIR) $(SCRIPTS_DIR) $(TESTS_DIR)
	$(ISORT) --check-only $(SRC_DIR) $(SCRIPTS_DIR) $(TESTS_DIR)

# Testing targets
test: ## Run tests
	$(PYTEST) $(TESTS_DIR) -v

test-cov: ## Run tests with coverage
	$(PYTEST) $(TESTS_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

test-fast: ## Run fast tests only
	$(PYTEST) $(TESTS_DIR) -v -m "not slow"

# MLOps targets
mlflow-server: ## Start MLflow server
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:./mlruns

dvc-init: ## Initialize DVC
	dvc init
	dvc remote add -d local ./dvc-storage

dvc-add-data: ## Add data to DVC
	dvc add $(DATA_DIR)/raw/variants.vcf
	dvc add $(DATA_DIR)/processed/

dvc-pipeline: ## Run DVC pipeline
	dvc repro

dvc-metrics: ## Show DVC metrics
	dvc metrics show

dvc-plots: ## Generate DVC plots
	dvc plots show

# Docker targets
docker-build: ## Build Docker image
	docker build -t gene-variant-analysis:latest -f docker/Dockerfile .

docker-run: ## Run Docker container
	docker run -p 8000:8000 -p 5000:5000 gene-variant-analysis:latest

docker-compose-up: ## Start services with docker-compose
	docker-compose -f docker/docker-compose.yml up --build

docker-compose-down: ## Stop services
	docker-compose -f docker/docker-compose.yml down

# Jupyter targets
jupyter: ## Start Jupyter Lab
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

notebook-convert: ## Convert notebooks to HTML
	jupyter nbconvert --to html notebooks/*.ipynb

# Documentation targets
docs: ## Generate documentation
	cd docs && make html

docs-serve: ## Serve documentation
	cd docs/_build/html && python -m http.server 8080

# Cleaning targets
clean: ## Clean temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

clean-data: ## Clean processed data
	rm -rf $(DATA_DIR)/processed/*
	rm -rf $(DATA_DIR)/interim/*

clean-models: ## Clean trained models
	rm -rf $(MODELS_DIR)/trained/*
	rm -rf $(MODELS_DIR)/artifacts/*

clean-all: clean clean-data clean-models ## Clean everything

# CI/CD targets
ci: check-format lint test ## Run CI checks

pre-commit: format lint test ## Run pre-commit checks

# Utility targets
requirements: ## Update requirements.txt
	pip-compile requirements.in

requirements-dev: ## Update requirements-dev.txt
	pip-compile requirements-dev.in

check-deps: ## Check for dependency vulnerabilities
	safety check

profile: ## Profile the training script
	$(PYTHON) -m cProfile -o profile.prof scripts/train.py --vcf $(DATA_DIR)/raw/variants.vcf

# Development targets
dev-setup: setup create-dirs ## Complete development setup
	@echo "Development environment setup complete!"

sample-data: ## Create sample data for testing
	$(PYTHON) scripts/create_sample_data.py --output $(DATA_DIR)/raw/

demo: sample-data train evaluate ## Run complete demo pipeline
	@echo "Demo pipeline completed successfully!"

.PHONY: help install install-dev setup create-dirs download-data prepare-data \
        train train-ensemble train-all evaluate predict \
        format lint check-format test test-cov test-fast \
        mlflow-server dvc-init dvc-add-data dvc-pipeline dvc-metrics dvc-plots \
        docker-build docker-run docker-compose-up docker-compose-down \
        jupyter notebook-convert docs docs-serve \
        clean clean-data clean-models clean-all \
        ci pre-commit requirements requirements-dev check-deps profile \
        dev-setup sample-data demo
