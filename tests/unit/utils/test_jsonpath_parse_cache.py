import threading

import pytest
from jsonpath_ng.exceptions import JsonPathParserError

from dynamiq.utils import jsonpath
from dynamiq.utils.jsonpath import filter_all, is_expression, mapper, parse


@pytest.fixture(autouse=True)
def empty_cache():
    parse.cache_clear()
    yield
    parse.cache_clear()


@pytest.fixture
def counted_parser(monkeypatch):
    """Counts how often jsonpath_ng actually builds a parser for an expression."""
    calls = []
    original = jsonpath.parse_jsonpath

    def counting(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(jsonpath, "parse_jsonpath", counting)
    return calls


def test_parse_builds_each_expression_once(counted_parser):
    data = {"input": {"output": {"query": "hello"}}}

    for _ in range(50):
        assert mapper(data, {"result": "$.input.output.query"}) == {"result": "hello"}

    assert counted_parser == ["$.input.output.query"]


def test_parse_returns_the_same_object_for_the_same_expression():
    assert parse("$.a.b") is parse("$.a.b")
    assert parse("$.a.b") is not parse("$.a.c")


def test_parse_results_stay_correct_when_reused_on_other_data():
    expression = "$.items[*].name"

    assert filter_all({"items": [{"name": "a"}, {"name": "b"}]}, expression) == ["a", "b"]
    assert filter_all({"items": [{"name": "c"}]}, expression) == ["c"]


def test_parse_does_not_cache_failures(counted_parser):
    for _ in range(3):
        with pytest.raises(JsonPathParserError):
            parse("$.[")
        assert not is_expression("$.[")

    assert counted_parser.count("$.[") == 6
    assert parse.cache_info().currsize == 0


def test_parse_is_shared_safely_across_threads():
    data = [{"value": i} for i in range(100)]
    errors = []

    def resolve(i):
        try:
            assert mapper({"item": data[i]}, {"value": "$.item.value"}) == {"value": i}
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=resolve, args=(i,)) for i in range(100)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert parse.cache_info().currsize == 1
