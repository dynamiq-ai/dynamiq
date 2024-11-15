install-dependencies-dev:
	poetry install --only dev --no-root

install-dependencies-examples:
	poetry install --only examples --no-root

install-dependencies-main:
	poetry install --only main --no-root

install-dependencies-all:
	poetry install --with examples --no-root

install-pre-commit:
	pre-commit install --install-hooks

install:
	poetry install --no-root
	make install-pre-commit

prepare:
	pre-commit run --all-files

lint: prepare

test-integration:
	pytest tests/integration

test-integration-with-creds:
	pytest tests/integration_with_creds

test-exclude-integration-with-creds:
	pytest tests --ignore=tests/integration_with_creds

test-unit:
	pytest tests/unit

test:
	pytest tests

test-cov:
	mkdir -p ./reports
	coverage run -m pytest --junitxml=./reports/test-results.xml tests
	coverage report --skip-empty
	coverage html -d ./reports/htmlcov --omit="*/test_*,*/tests.py"
	coverage xml -o ./reports/coverage.xml --omit="*/test_*,*/tests.py"

test-cov-exclude-integration-with-creds:
	mkdir -p ./reports
	coverage run -m pytest --junitxml=./reports/test-results.xml tests --ignore=tests/integration_with_creds
	coverage report --skip-empty
	coverage html -d ./reports/htmlcov --omit="*/test_*,*/tests.py"
	coverage xml -o ./reports/coverage.xml --omit="*/test_*,*/tests.py"

build-mkdocs:
	rm -rf mkdocs/
	python scripts/generate_mkdocs.py
	cp README.md mkdocs/index.md
	cp -rf docs/tutorials/ mkdocs/tutorials/
	mkdocs build

publish-mkdocs:
	make build-mkdocs
	mkdocs gh-deploy --force

run-mkdocs-locally:
	make build-mkdocs
	mkdocs serve
