.PHONY: build shell check-rocm test test-cpu docs lint clean wheel check-wheel install-wheel-test

IMAGE=gradslam-rocm:rocm7.2.2-torch2.7.1-py3.12

build:
	docker compose build

shell:
	docker compose run --rm gradslam bash

check-rocm:
	docker compose run --rm gradslam python scripts/check_rocm_stack.py

test:
	docker compose run --rm gradslam pytest -q

test-cpu:
	pytest -q

docs:
	docker compose run --rm gradslam bash -c "cd docs && make clean && make html"

lint:
	ruff check gradslam tests
	ruff format gradslam tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info site docs/_build

wheel:
	rm -rf build dist *.egg-info
	python -m build

check-wheel: wheel
	python -m twine check dist/*
	check-wheel-contents dist/*.whl

install-wheel-test: wheel
	python -m pip install --force-reinstall dist/*.whl
	python -c "import importlib.metadata as m; import gradslam; print(gradslam.__version__, m.version('opengradslam'))"
