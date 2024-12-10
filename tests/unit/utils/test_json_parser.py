import pytest

from dynamiq.utils.json_parser import parse_llm_json_output


def test_valid_json():
    response = '{"score": 1.0}'
    expected_output = {"score": 1.0}
    assert parse_llm_json_output(response) == expected_output


def test_json_with_extra_text():
    response = 'The result is:\n{\n  "score": 1.0\n}\nThank you.'
    expected_output = {"score": 1.0}
    assert parse_llm_json_output(response) == expected_output


def test_json_with_single_quotes():
    response = "{'score': 1.0}"
    expected_output = {"score": 1.0}
    assert parse_llm_json_output(response) == expected_output


def test_json_with_trailing_commas():
    response = """
    {
        "score": 1.0,
        "status": "success",
    }
    """
    expected_output = {"score": 1.0, "status": "success"}
    assert parse_llm_json_output(response) == expected_output


def test_json_with_single_quotes_and_trailing_commas():
    response = """
    {
        'score': 1.0,
        'status': 'success',
    }
    """
    expected_output = {"score": 1.0, "status": "success"}
    assert parse_llm_json_output(response) == expected_output


def test_json_embedded_in_markdown_code_block():
    response = """
    Here is the result:

    ```json
    {
        "score": 1.0,
        "status": "success"
    }
    ```

    Best regards.
    """
    expected_output = {"score": 1.0, "status": "success"}
    assert parse_llm_json_output(response) == expected_output


def test_invalid_json_cannot_be_fixed():
    response = "This is not JSON at all."
    with pytest.raises(ValueError) as excinfo:
        parse_llm_json_output(response)
    assert "Response from LLM is not valid JSON" in str(excinfo.value)


def test_no_json_object_in_response():
    response = "The results are inconclusive."
    with pytest.raises(ValueError) as excinfo:
        parse_llm_json_output(response)
    assert "Response from LLM is not valid JSON" in str(excinfo.value)


def test_complex_nested_json_with_errors():
    response = """
    {
        'user': {
            'id': 123,
            'name': 'John Doe',
            'roles': ['admin', 'editor',],
        },
        'active': True,
    }
    """
    expected_output = {"user": {"id": 123, "name": "John Doe", "roles": ["admin", "editor"]}, "active": True}
    assert parse_llm_json_output(response) == expected_output


def test_json_array_instead_of_object():
    response = """
    [
        {
            "score": 1.0
        },
        {
            "score": 2.0
        }
    ]
    """
    expected_output = [{"score": 1.0}, {"score": 2.0}]
    assert parse_llm_json_output(response) == expected_output


def test_json_with_comments():
    response = """
    {
        "score": 1.0,  // This is the score
        "status": "success"  // This is the status
    }
    """
    expected_output = {"score": 1.0, "status": "success"}
    assert parse_llm_json_output(response) == expected_output


def test_json_with_multiple_objects():
    response = """
    First result:
    {
        "score": 1.0
    }
    Second result:
    {
        "score": 2.0
    }
    """
    expected_output = {"score": 1.0}
    assert parse_llm_json_output(response) == expected_output


def test_parse_llm_json_output_with_nested_objects():
    response = """
    {
        "user": {
            "id": 123,
            "name": "Alice",
            "preferences": {
                "notifications": true,
                "theme": "dark"
            }
        }
    }
    """
    expected_output = {"user": {"id": 123, "name": "Alice", "preferences": {"notifications": True, "theme": "dark"}}}
    assert parse_llm_json_output(response) == expected_output
