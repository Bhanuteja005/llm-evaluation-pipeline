.PHONY: help install test run clean docker-build docker-run format lint

help:
	@echo "LLM Evaluation Pipeline - Available Commands:"
	@echo ""
	@echo "  make install        Install dependencies and setup environment"
	@echo "  make test          Run test suite"
	@echo "  make run           Run example evaluation"
	@echo "  make run-sample    Run on provided sample files"
	@echo "  make format        Format code with black and isort"
	@echo "  make lint          Run linters (flake8, mypy)"
	@echo "  make clean         Clean up generated files"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run in Docker"
	@echo "  make stats         Show evaluation statistics"
	@echo ""

install:
	@echo "Installing dependencies..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env file"; fi
	@echo "✓ Installation complete"

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=term-missing

run:
	@echo "Running example evaluation..."
	python examples/run_example.py

run-sample:
	@echo "Running evaluation on sample files..."
	python run_sample_evaluation.py

format:
	@echo "Formatting code..."
	black src/ tests/ examples/
	isort src/ tests/ examples/
	@echo "✓ Code formatted"

lint:
	@echo "Running linters..."
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	mypy src/ --ignore-missing-imports
	@echo "✓ Linting complete"

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache .mypy_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info
	rm -f evaluations.db
	@echo "✓ Cleanup complete"

docker-build:
	@echo "Building Docker image..."
	docker build -t llm-eval-pipeline .
	@echo "✓ Docker image built"

docker-run:
	@echo "Running in Docker..."
	docker run -v $(PWD)/../Json:/input -v $(PWD)/output:/output \
		llm-eval-pipeline evaluate \
		-c /input/sample-chat-conversation-01.json \
		-x /input/sample_context_vectors-01.json \
		-o /output/docker-results.json

stats:
	@echo "Evaluation statistics:"
	python -m src.cli stats

# Development shortcuts
dev-install:
	pip install -e .
	pip install pytest pytest-cov black flake8 mypy isort

watch-test:
	pytest-watch tests/ -v
