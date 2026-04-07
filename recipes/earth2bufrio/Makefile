.PHONY: lint format test test-cov docs

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=earth2bufrio --cov-report=term-missing --cov-fail-under=80

docs:
	sphinx-build -b html docs/ docs/_build/html
