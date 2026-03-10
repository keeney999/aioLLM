.PHONY: install test lint run clean

install:
	poetry install

test:
	poetry run pytest tests/ -v --cov=src.llm

lint:
	poetry run pre-commit run --all-files

run:
	poetry run uvicorn src.llm.api:app --reload

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
