# Makefile for mini-sklearn

# Set PYTHONPATH for all commands
export PYTHONPATH := $(shell pwd)

# Default target
.PHONY: help
help:
	@echo "Mini-sklearn Test Commands:"
	@echo ""
	@echo "  make test          - Run all tests"
	@echo "  make test-ab       - Run only A/B tests"
	@echo "  make test-negative - Run only error handling tests"
	@echo "  make test-splits   - Run only data splitting tests"
	@echo "  make test-minmax   - Run only MinMaxScaler tests"
	@echo "  make test-forest   - Run only RandomForest tests"
	@echo "  make test-quick    - Run tests with minimal output"
	@echo ""
	@echo "  make clean         - Clean up cache files"
	@echo "  make install       - Install dependencies"

# Test targets
.PHONY: test
test:
	@echo "🧪 Running all tests..."
	python3 -m pytest tests/ -v

.PHONY: test-ab
test-ab:
	@echo "🔄 Running A/B tests..."
	python3 -m pytest -v -k "ab"

.PHONY: test-negative
test-negative:
	@echo "❌ Running negative/error tests..."
	python3 -m pytest tests/test_negative_cases.py -v

.PHONY: test-splits
test-splits:
	@echo "✂️ Running data splitting tests..."
	python3 -m pytest tests/test_split_stratified_ab.py -v

.PHONY: test-minmax
test-minmax:
	@echo "📏 Running MinMaxScaler tests..."
	python3 -m pytest tests/test_minmax_ab.py -v

.PHONY: test-forest
test-forest:
	@echo "🌲 Running RandomForest tests..."
	python3 -m pytest tests/test_random_forest_ab.py -v

.PHONY: test-quick
test-quick:
	@echo "⚡ Running quick tests..."
	python3 -m pytest tests/ -q

# Utility targets
.PHONY: clean
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

.PHONY: install
install:
	@echo "📦 Installing dependencies..."
	pip3 install numpy pytest scikit-learn

# Development targets
.PHONY: format
format:
	@echo "🎨 Formatting code..."
	black mini_sklearn/ tests/ --line-length 88 || echo "Install black for formatting: pip install black"

.PHONY: lint
lint:
	@echo "🔍 Linting code..."
	ruff check mini_sklearn/ tests/ || echo "Install ruff for linting: pip install ruff"