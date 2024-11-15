from typing import Any


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
        # Handle logical operators with nested conditions
        if "operator" in filter_term and "conditions" in filter_term:
            operator = filter_term["operator"]
            if operator not in self.LOGICAL_OPERATORS:
                raise ValueError(f"Unsupported logical operator: {operator}")

            # Process each condition recursively
            sub_expressions = [self._parse_filter(cond) for cond in filter_term["conditions"]]
            return f"({self.LOGICAL_OPERATORS[operator].join(sub_expressions)})"

        # Handle comparison conditions
        elif "field" in filter_term and "operator" in filter_term and "value" in filter_term:
            field = filter_term["field"]
            operator = filter_term["operator"]
            value = filter_term["value"]

            # Build comparison expression
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

        if operator == "in" and isinstance(value, list):
            return f"{field} in {value}"
        elif operator == "not in" and isinstance(value, list):
            return f"{field} not in {value}"
        elif isinstance(value, str):
            return f'{field} {self.COMPARISON_OPERATORS[operator]} "{value}"'
        else:
            return f"{field} {self.COMPARISON_OPERATORS[operator]} {value}"

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
