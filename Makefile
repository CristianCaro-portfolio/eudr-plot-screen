.PHONY: install lint test demo fetch run

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

test:
	pytest -q

demo:
	python -m plotscreen.cli demo

fetch:
	python -m plotscreen.cli fetch --baseline --verbose

run:
	python -m plotscreen.cli run --verbose
