import json
import math
import re
from typing import Any

# A field name, or `$meta`, optionally followed by JSON path segments such as `["key"]` or `[0]`.
FIELD_PATTERN = re.compile(r'(?:\$meta|[A-Za-z_][A-Za-z0-9_]*)(?:\[(?:\d+|"[A-Za-z0-9_\-. ]*")\])*')


class Filter:
    LOGICAL_OPERATORS = {"AND": " and ", "OR": " or ", "NOT": "not "}
    COMPARISON_OPERATORS = {
        "==": "==",
        "!=": "!=",
        ">": ">",
        ">=": ">=",
        "<": "<",
        "<=": "<=",
        "in": "in",
        "not in": "not in",
        "contains_any": "contains_any",
    }

    def __init__(self, filter_criteria: dict[str, Any]):
        """
        Initializes the Filter object with filter criteria.

        Args:
            filter_criteria (Dict[str, Any]): The filters to apply.
        """
        self.filter_criteria = filter_criteria

    def build_filter_expression(self) -> str:
        """
        Builds the filter expression string from the filter criteria.

        Returns:
            str: The constructed filter expression string compatible with the database.
        """
        return self._parse_filter(self.filter_criteria)

    def _parse_filter(self, filter_term: dict[str, Any]) -> str:
        """
        Recursively parse the filter criteria to build a Milvus-compatible filter expression.

        Args:
            filter_term (Dict[str, Any]): The filter dictionary to parse.

        Returns:
            str: A Milvus-compatible filter expression.
        """
        if "operator" in filter_term and "conditions" in filter_term:
            operator = filter_term["operator"]
            if operator not in self.LOGICAL_OPERATORS:
                raise ValueError(f"Unsupported logical operator: {operator}")

            sub_expressions = [self._parse_filter(cond) for cond in filter_term["conditions"]]
            return f"({self.LOGICAL_OPERATORS[operator].join(sub_expressions)})"

        elif "field" in filter_term and "operator" in filter_term and "value" in filter_term:
            field = filter_term["field"]
            operator = filter_term["operator"]
            value = filter_term["value"]

            return self._build_comparison_expression(field, operator, value)

        else:
            raise ValueError("Invalid filter structure")

    def _build_comparison_expression(self, field: str, operator: str, value: Any) -> str:
        """
        Constructs a comparison expression based on field, operator, and value.

        Args:
            field (str): The field to filter on.
            operator (str): The comparison operator.
            value (Any): The value to compare against.

        Returns:
            str: A Milvus-compatible comparison expression.
        """
        if operator not in self.COMPARISON_OPERATORS:
            raise ValueError(f"Unsupported comparison operator: {operator}")

        # Field and value are written into the expression text, so both must not be able to change its structure.
        if not isinstance(field, str) or not FIELD_PATTERN.fullmatch(field):
            raise ValueError(f"Invalid filter field: {field!r}")

        if operator == "contains_any":
            if not isinstance(value, list):
                raise ValueError(f"{field}'s value must be a list when using 'contains_any' comparator")
            # dynamiq's MilvusVectorStore stores metadata as dynamic JSON fields, so the JSON
            # variant is used rather than ARRAY_CONTAINS_ANY (which targets typed Array fields).
            # https://milvus.io/docs/json-operators.md
            return f"json_contains_any({field}, {self._format_value(value)})"
        elif operator == "in" and isinstance(value, list):
            return f"{field} in {self._format_value(value)}"
        elif operator == "not in" and isinstance(value, list):
            return f"{field} not in {self._format_value(value)}"
        else:
            return f"{field} {self.COMPARISON_OPERATORS[operator]} {self._format_value(value)}"

    @classmethod
    def _format_value(cls, value: Any) -> str:
        """
        Renders a filter value as a Milvus expression literal.

        Args:
            value (Any): A string, number, boolean, or a list of those.

        Returns:
            str: The literal to place in the expression.

        Raises:
            ValueError: If the value has an unsupported type.
        """
        if isinstance(value, list):
            return f"[{', '.join(cls._format_scalar(v) for v in value)}]"
        return cls._format_scalar(value)

    @staticmethod
    def _format_scalar(value: Any) -> str:
        if isinstance(value, str):
            # A JSON string is a valid Milvus string literal: quotes and backslashes are escaped.
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return repr(value)
        raise ValueError(f"Unsupported filter value: {value!r}")

    @staticmethod
    def from_dict(filter_dict: dict[str, Any]) -> "Filter":
        """
        Creates a Filter instance from a dictionary.

        Args:
            filter_dict (Dict[str, Any]): Dictionary defining filter criteria.

        Returns:
            Filter: The Filter instance.
        """
        return Filter(filter_dict)
