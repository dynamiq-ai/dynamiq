FROM python:3.13.10-slim AS runtime

ENV PYTHONPATH=/app/
ENV UV_PROJECT_ENVIRONMENT=/usr/local
ENV UV_COMPILE_BYTECODE=1
ENV PATH="/opt/uv/bin:$PATH"

# Install uv from PyPI into an isolated venv (so `uv sync` into /usr/local can't
# remove it), avoiding a build-time dependency on ghcr.io.
RUN python -m venv /opt/uv && /opt/uv/bin/pip install --no-cache-dir uv==0.11.24

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl make build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./pyproject.toml /app/pyproject.toml
COPY ./uv.lock /app/uv.lock

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

FROM runtime AS develop

COPY ./.pre-commit-config.yaml /app/.pre-commit-config.yaml
COPY ./setup.cfg /app/setup.cfg
COPY ./Makefile /app/Makefile
COPY ./dynamiq /app/dynamiq
COPY ./examples /app/examples
COPY ./tests /app/tests
