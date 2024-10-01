FROM python:3.12.2-slim as runtime

ENV PYTHONPATH=/app/
ENV RUNTIME_PACKAGES="git curl make"
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VERSION=1.8.3
ENV POETRY_VIRTUALENVS_CREATE=false
ENV PATH=${POETRY_HOME}/bin:${PATH}

RUN apt-get update && apt-get install -y $RUNTIME_PACKAGES
RUN apt-get install build-essential -y
RUN curl -sSL https://install.python-poetry.org | python3 - --yes

WORKDIR /app

COPY ./pyproject.toml /app/pyproject.toml
COPY ./poetry.lock /app/poetry.lock
COPY ./Makefile /app/Makefile

RUN poetry install --no-root

FROM runtime AS develop

COPY ./.pre-commit-config.yaml /app/.pre-commit-config.yaml
COPY ./setup.cfg /app/setup.cfg
COPY ./dynamiq /app/dynamiq
COPY ./examples /app/examples
COPY ./tests /app/tests
