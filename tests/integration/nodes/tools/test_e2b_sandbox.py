"""Integration tests for E2BInterpreterTool file params injection (E2B_API_KEY required)."""

import io
import os

import pytest

from dynamiq.connections import E2B as E2BConnection
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterInputSchema, E2BInterpreterTool
from dynamiq.storages.file.base import FileInfo


@pytest.fixture(scope="module")
def e2b_tool():
    if not os.getenv("E2B_API_KEY"):
        pytest.skip("E2B_API_KEY is not set")
    tool = E2BInterpreterTool(
        connection=E2BConnection(),
        persistent_sandbox=True,
        is_optimized_for_agents=False,
    )
    yield tool
    tool.close()


@pytest.mark.integration
def test_params_with_files_and_scalars(e2b_tool):
    """All param types (BytesIO, FileInfo, list of files, str, int, dict) are injected and usable."""
    csv_file = io.BytesIO(b"name,score\nAlice,95\nBob,87\n")
    csv_file.name = "scores.csv"

    config = FileInfo(
        name="config.json",
        path="/tmp/config.json",
        size=20,
        content=b'{"factor": 10}',
        content_type="application/json",
    )

    extra1 = io.BytesIO(b"xxx")
    extra1.name = "e1.txt"
    extra2 = io.BytesIO(b"yy")
    extra2.name = "e2.txt"

    input_data = E2BInterpreterInputSchema(
        python=(
            "import csv, json\n"
            "with open(scores) as f:\n"
            "    count = len(list(csv.reader(f))) - 1\n"
            "with open(cfg) as f:\n"
            "    factor = json.load(f)['factor']\n"
            "extra_sizes = []\n"
            "for p in extras:\n"
            "    with open(p) as f:\n"
            "        extra_sizes.append(len(f.read()))\n"
            "result = count * factor * multiplier\n"
            "print(f'{label}: {result}, extras={extra_sizes}, mode={opts[\"mode\"]}')"
        ),
        params={
            "scores": csv_file,
            "cfg": config,
            "extras": [extra1, extra2],
            "multiplier": 5,
            "label": "Total",
            "opts": {"mode": "full", "verbose": True},
        },
    )
    result = e2b_tool.execute(input_data)
    output = result["content"]["code_execution"]
    assert "100" in output
    assert "Total" in output
    assert "[3, 2]" in output
    assert "mode=full" in output
