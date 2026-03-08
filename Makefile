# =============================================================================
# LLM Inference Service Makefile
# =============================================================================
# Common development and deployment commands

.PHONY: help install install-dev run run-dev build up up-build down logs logs-llm test lint format clean \
        up-monitoring up-full docker-status docker-shell redis-cli load-test load-test-headless

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
DOCKER := docker
DOCKER_COMPOSE := docker compose
IMAGE_NAME := llm-inference-service
IMAGE_TAG := latest

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "LLM Inference Service - Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  %-15s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# =============================================================================
# Installation
# =============================================================================

install: ## Install production dependencies
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev]"

# =============================================================================
# Local Development
# =============================================================================

run: ## Run the service locally
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000

run-dev: ## Run with hot reload for development
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# =============================================================================
# Docker Commands
# =============================================================================

build: ## Build Docker image
	$(DOCKER) build -t $(IMAGE_NAME):$(IMAGE_TAG) --target production .

build-dev: ## Build development Docker image
	$(DOCKER) build -t $(IMAGE_NAME):dev --target development .

up: ## Start LLM service + Redis
	$(DOCKER_COMPOSE) up -d
	@echo ""
	@echo "Services started:"
	@echo "  - LLM Service: http://localhost:8000"
	@echo "  - Redis:       localhost:6379"

up-build: ## Build and start services
	$(DOCKER_COMPOSE) up -d --build

down: ## Stop all services
	$(DOCKER_COMPOSE) down

logs: ## Follow service logs
	$(DOCKER_COMPOSE) logs -f

logs-llm: ## Follow LLM service logs only
	$(DOCKER_COMPOSE) logs -f llm

restart: ## Restart services
	$(DOCKER_COMPOSE) restart

# =============================================================================
# Testing
# =============================================================================

test: ## Run tests
	$(PYTHON) -m pytest tests/ -v

test-cov: ## Run tests with coverage
	$(PYTHON) -m pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

test-quick: ## Run tests without slow tests
	$(PYTHON) -m pytest tests/ -v -m "not slow"

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linter
	$(PYTHON) -m ruff check app/ tests/

lint-fix: ## Run linter with auto-fix
	$(PYTHON) -m ruff check app/ tests/ --fix

format: ## Format code
	$(PYTHON) -m ruff format app/ tests/

typecheck: ## Run type checker
	$(PYTHON) -m mypy app/

check: lint typecheck ## Run all code quality checks

# =============================================================================
# API Testing
# =============================================================================

test-health: ## Test health endpoint
	@curl -s http://localhost:8000/health | jq .

test-ready: ## Test readiness endpoint
	@curl -s http://localhost:8000/ready | jq .

test-models: ## Test models endpoint
	@curl -s http://localhost:8000/v1/models | jq .

test-completion: ## Test completion endpoint
	@curl -s http://localhost:8000/v1/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "prompt": "Hello, how are you?", "max_tokens": 50}' | jq .

test-chat: ## Test chat completion endpoint
	@curl -s http://localhost:8000/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}' | jq .

test-stream: ## Test streaming completion
	@curl -N http://localhost:8000/v1/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "prompt": "Count from 1 to 10:", "max_tokens": 100, "stream": true}'

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-docker: ## Clean up Docker resources
	$(DOCKER_COMPOSE) down -v --rmi local
	$(DOCKER) system prune -f

clean-all: clean clean-docker ## Clean everything

# =============================================================================
# Documentation
# =============================================================================

docs-serve: ## Serve API documentation locally
	@echo "API docs available at: http://localhost:8000/docs"
	@echo "ReDoc available at: http://localhost:8000/redoc"

# =============================================================================
# Utility
# =============================================================================

shell: ## Open Python shell with app context
	$(PYTHON) -c "from app.main import *; import code; code.interact(local=locals())"

env-template: ## Generate .env template
	@echo "# LLM Inference Service Configuration" > .env.example
	@echo "" >> .env.example
	@echo "# Model Settings" >> .env.example
	@echo "MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0" >> .env.example
	@echo "TENSOR_PARALLEL_SIZE=1" >> .env.example
	@echo "GPU_MEMORY_UTILIZATION=0.90" >> .env.example
	@echo "MAX_NUM_SEQS=256" >> .env.example
	@echo "DTYPE=auto" >> .env.example
	@echo "" >> .env.example
	@echo "# Server Settings" >> .env.example
	@echo "HOST=0.0.0.0" >> .env.example
	@echo "PORT=8000" >> .env.example
	@echo "DEBUG=false" >> .env.example
	@echo "" >> .env.example
	@echo "# HuggingFace" >> .env.example
	@echo "HF_TOKEN=" >> .env.example
	@echo "Generated .env.example"

version: ## Show version
	@$(PYTHON) -c "from app import __version__; print(__version__)"

# =============================================================================
# Docker Compose - Full Stack
# =============================================================================

up-monitoring: ## Start with Prometheus + Grafana monitoring
	$(DOCKER_COMPOSE) --profile monitoring up -d
	@echo ""
	@echo "Services started:"
	@echo "  - LLM Service: http://localhost:8000"
	@echo "  - Prometheus:  http://localhost:9090"
	@echo "  - Grafana:     http://localhost:3000 (admin/admin)"

up-full: ## Start all services (LLM + Redis + Prometheus + Grafana)
	$(DOCKER_COMPOSE) --profile full up -d
	@echo ""
	@echo "All services started:"
	@echo "  - LLM Service: http://localhost:8000"
	@echo "  - Redis:       localhost:6379"
	@echo "  - Prometheus:  http://localhost:9090"
	@echo "  - Grafana:     http://localhost:3000 (admin/admin)"

docker-status: ## Show Docker container status
	@echo "=== Containers ==="
	$(DOCKER_COMPOSE) ps
	@echo ""
	@echo "=== Resource Usage ==="
	$(DOCKER) stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

docker-shell: ## Open shell in LLM container
	$(DOCKER_COMPOSE) exec llm /bin/sh

redis-cli: ## Open Redis CLI
	$(DOCKER_COMPOSE) exec redis redis-cli

# =============================================================================
# Load Testing
# =============================================================================

load-test: ## Run Locust load test (requires: pip install locust)
	locust -f tests/load/locustfile.py --host http://localhost:8000

load-test-headless: ## Run headless load test (10 users, 60 seconds)
	locust -f tests/load/locustfile.py --host http://localhost:8000 --headless -u 10 -r 2 --run-time 60s
