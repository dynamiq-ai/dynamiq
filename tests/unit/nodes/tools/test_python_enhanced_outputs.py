"""
Test cases for enhanced Python tool structured output functionality.
Tests the new capability for Python tools to return structured data with content + metadata.
"""

import pytest

from dynamiq.nodes.agents.utils import process_tool_output_for_agent
from dynamiq.nodes.tools.python import Python
from dynamiq.runnables import RunnableResult, RunnableStatus


class TestPythonToolEnhancedOutputs:
    """Test enhanced Python tool output functionality."""

    def test_structured_output_with_content_key(self):
        """Test Python tool returns structured output with content key intact."""
        python_code = """
def run(input_data):
    return {
        'content': 'Main calculation result',
        'metadata': {'source': 'calculation', 'confidence': 0.95},
        'data': {'numbers': [1, 2, 3, 4, 5], 'sum': 15},
        'status': 'success',
        'timestamp': '2024-01-01T00:00:00Z'
    }
"""
        python_node = Python(code=python_code)
        result = python_node.run({})

        assert isinstance(result, RunnableResult)
        assert result.status == RunnableStatus.SUCCESS

        expected_output = {
            "content": "Main calculation result",
            "metadata": {"source": "calculation", "confidence": 0.95},
            "data": {"numbers": [1, 2, 3, 4, 5], "sum": 15},
            "status": "success",
            "timestamp": "2024-01-01T00:00:00Z",
        }
        assert result.output == expected_output

        assert set(result.output.keys()) == {"content", "metadata", "data", "status", "timestamp"}

        assert result.output["content"] == "Main calculation result"
        assert result.output["metadata"]["confidence"] == 0.95
        assert result.output["data"]["sum"] == 15

    def test_dict_output_without_content_key_backward_compatibility(self):
        """Test Python tool maintains backward compatibility for dict without content key."""
        python_code = """
def run(input_data):
    return {
        'result': 'calculation complete',
        'numbers': [1, 2, 3],
        'sum': 6,
        'metadata': {'method': 'addition'}
    }
"""
        python_node = Python(code=python_code)
        result = python_node.run({})

        assert isinstance(result, RunnableResult)
        assert result.status == RunnableStatus.SUCCESS

        assert "content" in result.output
        assert len(result.output) == 1

        inner_dict = result.output["content"]
        assert isinstance(inner_dict, dict)
        assert inner_dict["result"] == "calculation complete"
        assert inner_dict["numbers"] == [1, 2, 3]
        assert inner_dict["sum"] == 6
        assert inner_dict["metadata"]["method"] == "addition"

    def test_backward_compatibility_simple_return(self):
        """Test backward compatibility with simple non-dict returns."""
        python_code = """
def run(input_data):
    return 'simple string result'
"""
        python_node = Python(code=python_code)
        result = python_node.run({})

        assert isinstance(result, RunnableResult)
        assert result.status == RunnableStatus.SUCCESS
        assert result.output == {"content": "simple string result"}

    def test_backward_compatibility_number_return(self):
        """Test backward compatibility with number returns."""
        python_code = """
def run(input_data):
    return 42
"""
        python_node = Python(code=python_code)
        result = python_node.run({})

        assert isinstance(result, RunnableResult)
        assert result.status == RunnableStatus.SUCCESS
        assert result.output == {"content": 42}

    def test_backward_compatibility_list_return(self):
        """Test backward compatibility with list returns."""
        python_code = """
def run(input_data):
    return [1, 2, 3, 'test']
"""
        python_node = Python(code=python_code)
        result = python_node.run({})

        assert isinstance(result, RunnableResult)
        assert result.status == RunnableStatus.SUCCESS
        assert result.output == {"content": [1, 2, 3, "test"]}

    def test_agent_processing_structured_output(self):
        """Test how agent processing handles structured outputs."""
        structured_output = {
            "content": "This is the main result",
            "metadata": {"source": "calculation", "confidence": 0.95},
            "data": {"numbers": [1, 2, 3, 4, 5]},
            "status": "success",
        }

        processed = process_tool_output_for_agent(structured_output)

        assert processed == "This is the main result"

        assert structured_output["metadata"]["confidence"] == 0.95
        assert structured_output["data"]["numbers"] == [1, 2, 3, 4, 5]

    def test_agent_processing_dict_without_content(self):
        """Test agent processing of dict without content key (backward compatibility)."""
        dict_output = {"content": {"result": "calculation complete", "numbers": [1, 2, 3], "sum": 6}}

        processed = process_tool_output_for_agent(dict_output)

        assert isinstance(processed, str)
        assert "calculation complete" in processed
        assert "1" in processed and "2" in processed and "3" in processed  # JSON format may vary

    def test_complex_nested_structure(self):
        """Test complex nested data structures are handled correctly."""
        python_code = """
def run(input_data):
    return {
        'content': 'Analysis complete',
        'results': {
            'statistics': {
                'mean': 2.5,
                'median': 2.5,
                'std': 1.29
            },
            'data_points': [
                {'x': 1, 'y': 2, 'category': 'A'},
                {'x': 2, 'y': 3, 'category': 'B'},
                {'x': 3, 'y': 4, 'category': 'A'}
            ]
        },
        'metadata': {
            'version': '1.0',
            'algorithm': 'linear_analysis',
            'parameters': {'threshold': 0.05, 'iterations': 100}
        }
    }
"""
        python_node = Python(code=python_code)
        result = python_node.run({})

        assert isinstance(result, RunnableResult)
        assert result.status == RunnableStatus.SUCCESS

        assert result.output["content"] == "Analysis complete"
        assert result.output["results"]["statistics"]["mean"] == 2.5
        assert len(result.output["results"]["data_points"]) == 3
        assert result.output["results"]["data_points"][0]["category"] == "A"
        assert result.output["metadata"]["parameters"]["threshold"] == 0.05

    def test_empty_dict_return(self):
        """Test handling of empty dict return (backward compatibility)."""
        python_code = """
def run(input_data):
    return {}
"""
        python_node = Python(code=python_code)
        result = python_node.run({})

        assert isinstance(result, RunnableResult)
        assert result.status == RunnableStatus.SUCCESS
        assert result.output == {"content": {}}

    def test_dict_with_content_as_none(self):
        """Test dict with content key set to None."""
        python_code = """
def run(input_data):
    return {
        'content': None,
        'error': 'No result available',
        'status': 'failed'
    }
"""
        python_node = Python(code=python_code)
        result = python_node.run({})

        assert isinstance(result, RunnableResult)
        assert result.status == RunnableStatus.SUCCESS

        expected_output = {"content": None, "error": "No result available", "status": "failed"}
        assert result.output == expected_output

    def test_practical_example_data_analysis(self):
        """Test a practical data analysis example with structured output."""
        python_code = """
def run(input_data):
    numbers = input_data.get('numbers', [1, 2, 3, 4, 5])

    # Perform analysis
    total = sum(numbers)
    average = total / len(numbers)
    maximum = max(numbers)
    minimum = min(numbers)

    return {
        'content': f'Analyzed {len(numbers)} numbers: sum={total}, average={average:.2f}',
        'summary': {
            'count': len(numbers),
            'sum': total,
            'average': average,
            'max': maximum,
            'min': minimum
        },
        'raw_data': numbers,
        'metadata': {
            'analysis_type': 'basic_statistics',
            'timestamp': '2024-01-01T00:00:00Z'
        }
    }
"""
        python_node = Python(code=python_code)
        result = python_node.run({"numbers": [10, 20, 30, 40, 50]})

        assert isinstance(result, RunnableResult)
        assert result.status == RunnableStatus.SUCCESS

        assert "Analyzed 5 numbers" in result.output["content"]
        assert "sum=150" in result.output["content"]
        assert "average=30.00" in result.output["content"]

        assert result.output["summary"]["count"] == 5
        assert result.output["summary"]["sum"] == 150
        assert result.output["summary"]["average"] == 30.0
        assert result.output["raw_data"] == [10, 20, 30, 40, 50]
        assert result.output["metadata"]["analysis_type"] == "basic_statistics"

    def test_agent_optimized_mode_compatibility(self):
        """Test that is_optimized_for_agents mode still works with enhanced outputs."""
        python_code = """
def run(input_data):
    return {
        'content': 'Structured result for agent',
        'data': {'key': 'value'},
        'count': 42
    }
"""
        python_node = Python(code=python_code)
        python_node.is_optimized_for_agents = True

        result = python_node.run({})

        assert isinstance(result, RunnableResult)
        assert result.status == RunnableStatus.SUCCESS

        expected_output = {
            "content": "Structured result for agent",
            "data": {"key": "value"},
            "count": 42,
        }
        assert result.output == expected_output
        assert isinstance(result.output["content"], str)
        assert isinstance(result.output["data"], dict)
        assert isinstance(result.output["count"], int)

    def test_agent_optimized_mode_dict_without_content(self):
        """Test agent-optimized mode with dict that doesn't have content key."""
        python_code = """
def run(input_data):
    return {
        'result': 'success',
        'numbers': [1, 2, 3],
        'count': 42
    }
"""
        python_node = Python(code=python_code)
        python_node.is_optimized_for_agents = True

        result = python_node.run({})

        assert isinstance(result, RunnableResult)
        assert result.status == RunnableStatus.SUCCESS

        assert "content" in result.output
        assert len(result.output) == 1
        assert isinstance(result.output["content"], str)
        assert "success" in result.output["content"]
        assert "[1, 2, 3]" in result.output["content"] or "1" in result.output["content"]


@pytest.mark.parametrize(
    ("return_data", "expected_behavior"),
    [
        ({"content": "test", "extra": "data"}, "structured"),
        ({"result": "success", "count": 5}, "wrapped"),
        ({}, "wrapped"),
        ("simple result", "wrapped"),
    ],
)
def test_output_handling_parametrized(return_data, expected_behavior):
    """Parametrized test for different output handling scenarios."""
    python_code = f"""
def run(input_data):
    return {return_data!r}
"""
    python_node = Python(code=python_code)
    result = python_node.run({})

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    if expected_behavior == "structured":
        assert result.output == return_data
        assert "content" in result.output
    else:
        assert "content" in result.output
        assert len(result.output) == 1
        assert result.output["content"] == return_data
