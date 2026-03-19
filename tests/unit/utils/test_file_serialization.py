"""Unit tests for serialize_file, serialize_files_in_value, and AgentToolResultEventMessageData.to_dict."""

from io import BytesIO
from typing import Any

from pydantic import BaseModel

from dynamiq.types.streaming import AgentToolData, AgentToolResultEventMessageData
from dynamiq.utils.utils import serialize_file, serialize_files_in_value


class TestSerializeFile:
    def test_bytesio_with_attributes(self):
        bio = BytesIO(b"hello")
        bio.name = "test.txt"
        bio.content_type = "text/plain"

        result = serialize_file(bio)

        assert result["content"] == "aGVsbG8="
        assert result["size"] == 5
        assert result["name"] == "test.txt"
        assert result["mime_type"] == "text/plain"

    def test_bytesio_without_attributes(self):
        bio = BytesIO(b"data")
        result = serialize_file(bio)

        assert result["content"] == "ZGF0YQ=="
        assert result["size"] == 4
        assert "name" not in result
        assert "mime_type" not in result

    def test_raw_bytes(self):
        result = serialize_file(b"raw")
        assert result["content"] == "cmF3"
        assert result["size"] == 3

    def test_non_file_passthrough(self):
        assert serialize_file("just a string") == "just a string"
        assert serialize_file(42) == 42
        assert serialize_file(None) is None


class TestSerializeFilesInValue:
    def test_flat_bytesio(self):
        bio = BytesIO(b"x")
        result = serialize_files_in_value(bio)
        assert isinstance(result, dict)
        assert result["content"] == "eA=="

    def test_dict_with_nested_files(self):
        bio = BytesIO(b"inner")
        data = {"key": "value", "file": bio}
        result = serialize_files_in_value(data)

        assert result["key"] == "value"
        assert isinstance(result["file"], dict)
        assert result["file"]["size"] == 5

    def test_list_of_files(self):
        files = [BytesIO(b"a"), BytesIO(b"bb")]
        result = serialize_files_in_value(files)

        assert len(result) == 2
        assert result[0]["size"] == 1
        assert result[1]["size"] == 2

    def test_deeply_nested(self):
        data = {"level1": {"level2": [{"file": BytesIO(b"deep")}]}}
        result = serialize_files_in_value(data)
        assert result["level1"]["level2"][0]["file"]["size"] == 4

    def test_tuple_preserved(self):
        data = (BytesIO(b"a"), "text")
        result = serialize_files_in_value(data)

        assert isinstance(result, tuple)
        assert isinstance(result[0], dict)
        assert result[0]["size"] == 1
        assert result[1] == "text"

    def test_primitives_unchanged(self):
        assert serialize_files_in_value("hello") == "hello"
        assert serialize_files_in_value(42) == 42
        assert serialize_files_in_value(None) is None
        assert serialize_files_in_value(True) is True

    def test_pydantic_model_with_file(self):
        class Inner(BaseModel):
            attachment: Any = None

        bio = BytesIO(b"model-file")
        bio.name = "doc.pdf"
        bio.content_type = "application/pdf"

        result = serialize_files_in_value(Inner(attachment=bio))

        assert isinstance(result, dict)
        assert result["attachment"]["size"] == 10
        assert result["attachment"]["name"] == "doc.pdf"

    def test_nested_model_inside_dict(self):
        class Wrapper(BaseModel):
            file: Any = None

        bio = BytesIO(b"nested")
        result = serialize_files_in_value({"outer": Wrapper(file=bio)})

        assert result["outer"]["file"]["size"] == 6

    def test_model_without_files_passthrough(self):
        class Plain(BaseModel):
            name: str = "test"
            count: int = 0

        result = serialize_files_in_value(Plain())
        assert result == {"name": "test", "count": 0}


class TestAgentToolResultToDict:
    def _make_model(self, **overrides):
        defaults = {
            "tool_run_id": "run-1",
            "name": "TestTool",
            "tool": AgentToolData(name="TestTool", type="test.Tool"),
            "input": {"prompt": "test"},
            "result": "ok",
            "files": [],
            "loop_num": 1,
        }
        defaults.update(overrides)
        return AgentToolResultEventMessageData(**defaults)

    def test_files_serialized(self):
        bio = BytesIO(b"img")
        bio.name = "image.png"
        bio.content_type = "image/png"

        model = self._make_model(files=[bio])
        data = model.to_dict()

        assert len(data["files"]) == 1
        f = data["files"][0]
        assert isinstance(f, dict)
        assert f["content"] == "aW1n"
        assert f["size"] == 3
        assert f["name"] == "image.png"
        assert f["mime_type"] == "image/png"

    def test_output_with_nested_file(self):
        bio = BytesIO(b"nested")
        bio.name = "report.pdf"
        bio.content_type = "application/pdf"

        model = self._make_model(output={"info": {"file": bio}})
        data = model.to_dict()

        assert data["output"]["info"]["file"]["size"] == 6
        assert data["output"]["info"]["file"]["name"] == "report.pdf"

    def test_result_with_file(self):
        bio = BytesIO(b"res")
        model = self._make_model(result={"data": bio})
        data = model.to_dict()

        assert data["result"]["data"]["size"] == 3

    def test_input_with_file(self):
        bio = BytesIO(b"inp")
        model = self._make_model(input={"attachment": bio})
        data = model.to_dict()

        assert data["input"]["attachment"]["size"] == 3

    def test_no_files_passthrough(self):
        model = self._make_model(result="plain text", output={"key": "value"})
        data = model.to_dict()

        assert data["result"] == "plain text"
        assert data["output"] == {"key": "value"}
        assert data["files"] == []
