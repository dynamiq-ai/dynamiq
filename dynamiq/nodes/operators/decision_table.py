import re
from collections.abc import Callable
from typing import Any, ClassVar, Literal, NamedTuple
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, PrivateAttr

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.types import DecisionAggregation, DecisionHitPolicy, DecisionRule, NamedField
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.utils import TRUNCATE_LIST_LIMIT

CellLiteral = str | int | float | bool
Condition = Callable[[Any], bool]

MATCHED_RULES_KEY = "matched_rules"
NUMERIC_TYPES = frozenset({"int", "float"})
COLUMN_TYPES = frozenset({"Any", "string", "bool"} | NUMERIC_TYPES)

# A number in decimal or exponent form: a YAML source holds a small float as `1e-05`, which is how Python
# writes it back, and a spreadsheet exports one the same way.
_NUMBER_TEXT = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_NUMBER = re.compile(rf"^{_NUMBER_TEXT}$")
_RANGE = re.compile(rf"^([\[(])\s*({_NUMBER_TEXT})\s*\.\.\s*({_NUMBER_TEXT})\s*([\])])$")
_COMPARISON = re.compile(r"^(==|!=|>=|<=|>|<|=)\s*(.*)$", re.DOTALL)
_ORDERINGS: dict[str, Callable[[Any, Any], bool]] = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
}
_FOLDS: dict[DecisionAggregation, Callable[[list[int | float]], int | float]] = {
    DecisionAggregation.SUM: sum,
    DecisionAggregation.MIN: min,
    DecisionAggregation.MAX: max,
}


def _number(text: str) -> int | float:
    return float(text) if any(char in text for char in ".eE") else int(text)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _equals(a: Any, b: Any) -> bool:
    # Python treats True == 1; a rule author does not.
    return isinstance(a, bool) == isinstance(b, bool) and a == b


def _comparable(a: Any, b: Any) -> bool:
    return (_is_number(a) and _is_number(b)) or (isinstance(a, str) and isinstance(b, str))


def cell_text(cell: Any) -> str:
    """A cell as the text the grid shows: YAML may hold a number or a boolean where the UI holds text."""
    if cell is None:
        return ""
    if isinstance(cell, bool):
        return "true" if cell else "false"
    return str(cell)


def parse_literal(text: str) -> CellLiteral | None:
    """A quoted value stays text, so `"700"` is not a number; `true` and `false` are booleans."""
    text = text.strip()
    if not text:
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    if text == "true":
        return True
    if text == "false":
        return False
    if _NUMBER.match(text):
        return _number(text)
    return text


def read_literal(text: str, column_type: str, where: str) -> CellLiteral:
    """A cell literal read as the column type; a literal the column cannot hold is a configuration error."""
    text = text.strip()
    if column_type == "string":
        literal = parse_literal(text)
        return literal if isinstance(literal, str) else text
    if column_type in NUMERIC_TYPES:
        if not _NUMBER.match(text):
            raise ValueError(f"{where}: expected a number, got {text!r}")
        return _number(text)
    if column_type == "bool":
        if text.lower() not in ("true", "false"):
            raise ValueError(f"{where}: expected true or false, got {text!r}")
        return text.lower() == "true"
    literal = parse_literal(text)
    if literal is None:
        raise ValueError(f"{where}: a value is required")
    return literal


def coerce_value(value: Any, column_type: str) -> Any:
    """The incoming value read as the column type. None means it cannot be, and only an empty cell matches."""
    if value is None or column_type == "Any":
        return value
    if column_type in NUMERIC_TYPES:
        if _is_number(value):
            return value
        if isinstance(value, str) and _NUMBER.match(value.strip()):
            return _number(value.strip())
        return None
    if column_type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if _is_number(value):
        return str(value)
    return None


def split_alternatives(text: str) -> list[str]:
    """Splits on the commas outside quotes, so `"a, b", c` is two alternatives.

    A quote opens an alternative only at its start, so an apostrophe inside a word (`O'Brien, Smith`)
    is text and the comma after it still splits.
    """
    parts: list[str] = []
    current = ""
    quote: str | None = None
    at_start = True
    for char in text:
        if quote:
            if char == quote:
                quote = None
            current += char
        elif char in "\"'" and at_start:
            quote = char
            current += char
            at_start = False
        elif char == ",":
            parts.append(current)
            current = ""
            at_start = True
        else:
            current += char
            if not char.isspace():
                at_start = False
    parts.append(current)
    return parts


def _any(value: Any) -> bool:
    return True


def _read_alternatives(alternatives: list[str], column_type: str, where: str) -> list[CellLiteral]:
    if any(not alternative.strip() for alternative in alternatives):
        raise ValueError(f"{where}: a list cannot contain an empty alternative")
    return [read_literal(alternative, column_type, where) for alternative in alternatives]


def compile_condition(text: str, column_type: str, where: str) -> Condition:
    """Turns one condition cell into a predicate over the coerced input value.

    The grammar is the one the editor validates: empty or `*` matches anything; a literal means equals;
    `>= 620`, `< 0.8`, `!= VA` compare; `[620..680]` is a range, `[` inclusive and `(` exclusive per
    side; a comma-separated list means any of, and `!= FHA, VA` none of. A value the column cannot
    read (None) matches only an empty cell, so a missing input never satisfies a condition, `!=` included.
    """
    text = text.strip()
    if not text or text == "*":
        return _any

    if bounds := _RANGE.match(text):
        open_bracket, low_text, high_text, close_bracket = bounds.groups()
        if column_type not in NUMERIC_TYPES and column_type != "Any":
            raise ValueError(f"{where}: a range needs a numeric column")
        low, high = _number(low_text), _number(high_text)
        if low > high:
            raise ValueError(f"{where}: range {text} runs backwards")
        include_low, include_high = open_bracket == "[", close_bracket == "]"

        def in_range(value: Any) -> bool:
            return (
                _is_number(value)
                and (low <= value if include_low else low < value)
                and (value <= high if include_high else value < high)
            )

        return in_range

    if comparison := _COMPARISON.match(text):
        symbol, rest = comparison.groups()
        if not rest.strip():
            raise ValueError(f"{where}: {symbol} needs a value to compare with")
        alternatives = split_alternatives(rest)
        if len(alternatives) > 1:
            # `!= FHA, VA` is none of the alternatives and `== FHA, VA` any of them. An ordering against
            # a list has no meaning, and read as one literal it would silently match every input.
            if symbol not in ("=", "==", "!="):
                raise ValueError(f"{where}: {symbol} takes one value, not a list")
            literals = _read_alternatives(alternatives, column_type, where)
            if symbol == "!=":
                return lambda value: value is not None and not any(_equals(value, literal) for literal in literals)
            return lambda value: value is not None and any(_equals(value, literal) for literal in literals)
        literal = read_literal(rest, column_type, where)
        if symbol in ("=", "=="):
            return lambda value: value is not None and _equals(value, literal)
        if symbol == "!=":
            return lambda value: value is not None and not _equals(value, literal)
        ordering = _ORDERINGS[symbol]
        return lambda value: _comparable(value, literal) and ordering(value, literal)

    alternatives = split_alternatives(text)
    if len(alternatives) > 1:
        literals = _read_alternatives(alternatives, column_type, where)
        return lambda value: value is not None and any(_equals(value, literal) for literal in literals)

    literal = read_literal(text, column_type, where)
    return lambda value: value is not None and _equals(value, literal)


class CompiledRule(NamedTuple):
    rule: DecisionRule
    conditions: list[Condition]
    outputs: list[CellLiteral | None]


class DecisionTableInputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class DecisionTable(Node):
    """Matches named inputs against rows of rules and returns the outputs of the matching rows.

    Inputs arrive under the input column names. Each rule holds one condition cell per input column
    (see `compile_condition` for the grammar) and one literal cell per output column, read as the
    column type; an empty output cell is None. Rules are compiled once, when the node is created,
    so a malformed cell fails then rather than on a run.

    Hit policies: `first` returns the first matching rule in table order, `unique` allows at most one
    match and fails the run on overlap, `collect` takes every match and folds each output column with
    the aggregation: `list` keeps the values, `count` is the number of matches, and `sum`, `min`, `max`
    fold a numeric column (or an Any column whose output cells are all numbers, judged over the table so
    a run's shape never depends on which rows matched) and otherwise keep the list. No match yields None
    for a folded column and an empty list for a collected one. The output always
    carries `matched_rules`, the `{id, name}` of the rules that fired, in table order; a rule switched
    off never fires.
    """

    name: str | None = "decision_table"
    group: Literal[NodeGroup.OPERATORS] = NodeGroup.OPERATORS
    hit_policy: DecisionHitPolicy = DecisionHitPolicy.FIRST
    aggregation: DecisionAggregation = DecisionAggregation.LIST
    input_columns: list[NamedField] = []
    output_columns: list[NamedField] = []
    rules: list[DecisionRule] = []
    input_schema: ClassVar[type[DecisionTableInputSchema]] = DecisionTableInputSchema

    _compiled: list[CompiledRule] = PrivateAttr(default_factory=list)
    _folded: list[bool] = PrivateAttr(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._compiled = self._compile()
        self._folded = [self._folds_numbers(index, column) for index, column in enumerate(self.output_columns)]

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"rules": True}

    def to_dict(self, include_secure_params: bool = True, for_tracing: bool = False, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        A trace keeps the first rules and the total: the rules that fired are in the output, and a large
        table would otherwise be copied into every run it takes part in.
        """
        data = super().to_dict(include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs)
        rules = self.rules[:TRUNCATE_LIST_LIMIT] if for_tracing else self.rules
        data["rules"] = [rule.model_dump(**kwargs) for rule in rules]
        if for_tracing:
            data["rules_count"] = len(self.rules)
        return data

    def _compile(self) -> list[CompiledRule]:
        for column in self.input_columns + self.output_columns:
            if column.type not in COLUMN_TYPES:
                raise ValueError(
                    f"Decision table '{self.name}': column '{column.name}' has unknown type {column.type!r}"
                )
        if any(column.name == MATCHED_RULES_KEY for column in self.output_columns):
            raise ValueError(
                f"Decision table '{self.name}': '{MATCHED_RULES_KEY}' is reserved for the rules that fired"
            )
        # The output is keyed by column name, so a name used twice would keep one value and drop the other silently.
        for side, columns in (("input", self.input_columns), ("output", self.output_columns)):
            names = [column.name for column in columns]
            if repeated := next((name for name in names if names.count(name) > 1), None):
                raise ValueError(f"Decision table '{self.name}': {side} column '{repeated}' is declared twice")

        compiled = []
        for index, rule in enumerate(self.rules, start=1):
            if not rule.enabled:
                continue
            label = f"Decision table '{self.name}', rule {index}" + (f" ({rule.name})" if rule.name else "")
            if len(rule.when) != len(self.input_columns) or len(rule.then) != len(self.output_columns):
                raise ValueError(f"{label}: expected one cell per input and per output column")
            conditions = [
                compile_condition(cell_text(cell), column.type, f"{label}, input '{column.name}'")
                for cell, column in zip(rule.when, self.input_columns)
            ]
            outputs = [
                (
                    read_literal(cell_text(cell), column.type, f"{label}, output '{column.name}'")
                    if cell_text(cell).strip()
                    else None
                )
                for cell, column in zip(rule.then, self.output_columns)
            ]
            compiled.append(CompiledRule(rule=rule, conditions=conditions, outputs=outputs))
        return compiled

    def execute(self, input_data: DecisionTableInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """Evaluates the rules against the input and returns the outputs of the matching rules."""
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **{**kwargs, "parent_run_id": kwargs.get("run_id", uuid4())})

        matched = self._match(input_data.model_dump())
        output: dict[str, Any] = {
            column.name: self._fold(index, [compiled.outputs[index] for compiled in matched])
            for index, column in enumerate(self.output_columns)
        }
        output[MATCHED_RULES_KEY] = [{"id": compiled.rule.id, "name": compiled.rule.name} for compiled in matched]
        return output

    def _match(self, values: dict[str, Any]) -> list[CompiledRule]:
        coerced = [coerce_value(values.get(column.name), column.type) for column in self.input_columns]
        matched = []
        for compiled in self._compiled:
            if all(condition(value) for condition, value in zip(compiled.conditions, coerced)):
                matched.append(compiled)
                if self.hit_policy == DecisionHitPolicy.FIRST:
                    break
        if self.hit_policy == DecisionHitPolicy.UNIQUE and len(matched) > 1:
            names = ", ".join(compiled.rule.name or compiled.rule.id for compiled in matched)
            raise ValueError(f"Decision table '{self.name}': rules {names} all match, but a unique table allows one")
        return matched

    def _fold(self, index: int, outputs: list[CellLiteral | None]) -> Any:
        if self.hit_policy != DecisionHitPolicy.COLLECT:
            return outputs[0] if outputs else None
        if self.aggregation == DecisionAggregation.LIST:
            return outputs
        if self.aggregation == DecisionAggregation.COUNT:
            return len(outputs)
        if not self._folded[index]:
            return outputs
        numbers = [value for value in outputs if _is_number(value)]
        if not numbers:
            return None
        return _FOLDS[self.aggregation](numbers)

    def _folds_numbers(self, index: int, column: NamedField) -> bool:
        """Whether an output column folds to a number under sum, min or max.

        Decided over the table's own cells rather than the rows that matched, so a column's shape is the
        same on every run and no match reads None whether the column is typed or left as Any.
        """
        if column.type in NUMERIC_TYPES:
            return True
        if column.type != "Any":
            return False
        cells = (compiled.outputs[index] for compiled in self._compiled)
        return all(_is_number(cell) for cell in cells if cell is not None)
