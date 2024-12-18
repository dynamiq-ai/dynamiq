import json

import pytest

from dynamiq.utils.json_parser import clean_json_string, parse_llm_json_output


def test_basic_single_quoted_keys_and_strings():
    input_str = "{'key': 'value'}"
    cleaned = clean_json_string(input_str)
    data = json.loads(cleaned)
    assert data == {"key": "value"}


def test_apostrophe_in_text():
    input_str = """
    {
       'message': 'Hello, I\\'ve got apples'
    }
    """
    cleaned = clean_json_string(input_str)
    data = json.loads(cleaned)
    assert data == {"message": "Hello, I've got apples"}


def test_comments_and_python_literals():
    input_str = """
    {
      // single line comment
      'bool_val': True, /* multi-line
      comment */
      'none_val': None,
      'another': 'Text with // a comment inside the string is not a comment'
    }
    """
    cleaned = clean_json_string(input_str)
    data = json.loads(cleaned)
    assert data == {
        "bool_val": True,
        "none_val": None,
        "another": "Text with // a comment inside the string is not a comment",
    }


def test_trailing_comma():
    input_str = """
    {
       'key1': 'v1',
       'key2': 'v2',
    }
    """
    cleaned = clean_json_string(input_str)
    data = json.loads(cleaned)
    assert data == {"key1": "v1", "key2": "v2"}


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


def test_json_array_in_text():
    response = """
    Here is your JSON:
    [
        {"answer": "text"},
        {"answer": "another text"}
    ]
    """
    expected_output = [{"answer": "text"}, {"answer": "another text"}]
    assert parse_llm_json_output(response) == expected_output


def test_mismatched_brackets():
    response = "Here is JSON: { 'key': 123 ] end"
    # Should fail due to mismatched '{' with ']'
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_llm_json_output(response)


def test_nested_comments():
    response = """
    {
      // This is a outer comment
      "example": "Here is // inside a string", /* Another comment */
      "other": "/* not a comment inside string */"
    }
    """
    # Comments outside strings should be removed,
    # but the sequences inside the string remain untouched.
    expected_output = {
        "example": "Here is // inside a string",
        "other": "/* not a comment inside string */",
    }
    assert parse_llm_json_output(response) == expected_output


def test_multiple_issues_single_quotes_trailing_comma_python_literals():
    response = """
    {
      'test': True, 'strings': ['yes', 'no',],
      'noneVal': None, // comment
    }
    """
    # All issues should be corrected by clean_json_string
    expected_output = {"test": True, "strings": ["yes", "no"], "noneVal": None}
    assert parse_llm_json_output(response) == expected_output


def test_empty_json_object_and_array():
    # Just confirm that empty objects/arrays parse without error.
    assert parse_llm_json_output("{}") == {}
    assert parse_llm_json_output("[]") == []
