install-dependencies-dev:
	uv sync --no-install-project --only-group dev

install-dependencies-examples:
	uv sync --no-install-project --only-group examples

install-dependencies-main:
	uv sync --no-install-project --no-default-groups

install-dependencies-all:
	uv sync --no-install-project --group examples

install-pre-commit:
	uv run pre-commit install --install-hooks

install:
	uv sync --no-install-project
	make install-pre-commit

prepare:
	uv run pre-commit run --all-files

lint: prepare

lock:
	uv lock

lock-upgrade:
	uv lock --upgrade

test-integration:
	uv run pytest tests/integration

test-integration-with-creds:
	uv run pytest tests/integration_with_creds -m "not smoke"

# Smoke tests: slow, paid end-to-end production scenarios. Excluded from every default target and
# run ONLY here (gated behind the `run-smoke-tests` PR label in CI). -n auto parallelizes the
# matrix; if E2B quota / provider 429s appear, dial back to `-n 4` or group same-provider cases.
test-smoke:
	uv run pytest tests -m smoke -n auto --dist worksteal

test-exclude-integration-with-creds:
	uv run pytest tests --ignore=tests/integration_with_creds -m "not smoke"

test-unit:
	uv run pytest tests/unit

test:
	uv run pytest tests -m "not smoke"

test-cov:
	mkdir -p ./reports
	uv run coverage run -m pytest --junitxml=./reports/test-results.xml -m "not smoke" tests
	uv run coverage report --skip-empty --skip-covered
	uv run coverage html -d ./reports/htmlcov --omit="*/test_*,*/tests.py"
	uv run coverage xml -o ./reports/coverage.xml --omit="*/test_*,*/tests.py"

test-cov-exclude-integration-with-creds:
	mkdir -p ./reports
	uv run coverage run -m pytest --junitxml=./reports/test-results.xml -m "not smoke" tests --ignore=tests/integration_with_creds
	uv run coverage report --skip-empty --skip-covered
	uv run coverage html -d ./reports/htmlcov --omit="*/test_*,*/tests.py"
	uv run coverage xml -o ./reports/coverage.xml --omit="*/test_*,*/tests.py"

build-mkdocs:
	rm -rf mkdocs/
	uv run python scripts/generate_mkdocs.py
	cp README.md mkdocs/index.md
	cp -rf docs/tutorials/ mkdocs/tutorials/
	uv run mkdocs build

publish-mkdocs:
	make build-mkdocs
	uv run mkdocs gh-deploy --force

run-mkdocs-locally:
	make build-mkdocs
	uv run mkdocs serve
