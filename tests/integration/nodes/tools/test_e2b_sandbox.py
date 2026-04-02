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
            "# mixed_list: ['hello', '/path/to/e1.txt', 42]\n"
            "assert isinstance(mixed_list, list) and len(mixed_list) == 3\n"
            "assert mixed_list[0] == 'hello'\n"
            "assert isinstance(mixed_list[1], str) and mixed_list[1].endswith('.txt')\n"
            "assert mixed_list[2] == 42\n"
            "# file_dict: {'data': '/path/to/scores.csv', 'threshold': 0.5}\n"
            "assert isinstance(file_dict, dict)\n"
            "assert isinstance(file_dict['data'], str) and file_dict['data'].endswith('.csv')\n"
            "assert file_dict['threshold'] == 0.5\n"
            "with open(file_dict['data']) as f:\n"
            "    dict_file_lines = len(f.readlines())\n"
            "print(json.dumps({'result': result, 'extras': extra_sizes, 'label': label, "
            "'mode': opts['mode'], 'mixed_list': mixed_list, 'dict_file_lines': dict_file_lines}))"
        ),
        params={
            "scores": csv_file,
            "cfg": config,
            "extras": [extra1, extra2],
            "multiplier": 5,
            "label": "Total",
            "opts": {"mode": "full", "verbose": True},
            "mixed_list": ["hello", extra1, 42],
            "file_dict": {"data": csv_file, "threshold": 0.5},
        },
    )
    result = e2b_tool.execute(input_data)
    output = result["content"]["code_execution"]
    assert isinstance(output, dict)
    assert output["result"] == 100
    assert output["label"] == "Total"
    assert output["extras"] == [3, 2]
    assert output["mode"] == "full"
    assert output["mixed_list"][0] == "hello"
    assert output["mixed_list"][1].endswith(".txt")
    assert output["mixed_list"][2] == 42
    assert output["dict_file_lines"] == 3
