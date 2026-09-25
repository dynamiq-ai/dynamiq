from collections.abc import Callable
from typing import Any, ClassVar, Literal
from uuid import uuid4

from jinja2 import TemplateSyntaxError
from pydantic import BaseModel, ConfigDict, PrivateAttr

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.operators.rules import (
    Reads,
    RecordSandbox,
    concrete,
    read_paths,
    refuse_clash,
    refuse_reserved_read,
    scope_for,
)
from dynamiq.nodes.types import ExpressionItem, NamedField
from dynamiq.runnables import RunnableConfig

# One sandbox for every Expression node: it keeps no state, it refuses attribute access that would
# reach Python internals, so an expression cannot escape into the process, and it reads a record's key
# before a method of the same name, as a rule does. The sandbox carries the Rules node's helpers, which
# serve an expression too: a due date or an age is a date computation over the same inputs.
_ENVIRONMENT = RecordSandbox()


class ExpressionInputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class Expression(Node):
    """Computes new fields from named inputs with Jinja2 expressions.

    Each expression is the body of a template variable without the braces (`price * quantity`,
    `'A' if score < 0.5 else 'B'`, `amount | round(2)`), compiled once in a sandboxed environment and
    evaluated with the inputs as its names. It returns the value itself, not rendered text. An input
    referred to on its own that is missing evaluates to None; using a missing input in arithmetic
    fails the run, as does an expression that reaches for Python internals. The helpers a rule can
    call are available as well: `has`, `days_between`, `date`, `today`, `len`, `abs`, `min`, `max`,
    `sum`, `round`, `text`, `number` and `first_present`; so are the tests `is present` and `is blank`.
    An input named like one of the helpers is the input where an expression reads it as a value and
    the helper where an expression calls it. A blank result, `text('  ')` or `date('')` say, comes out
    as None, like a missing input; a value `number()` or `date()` cannot read fails the run instead,
    `number('TBD')` as `date('March')` does. The output holds one key per expression, plus every input
    when `pass_through` is set, with expressions winning on a clash.
    """

    name: str | None = "expression"
    group: Literal[NodeGroup.OPERATORS] = NodeGroup.OPERATORS
    input_fields: list[NamedField] = []
    expressions: list[ExpressionItem] = []
    pass_through: bool = False
    input_schema: ClassVar[type[ExpressionInputSchema]] = ExpressionInputSchema

    _compiled: list[tuple[str, Callable[..., Any], Reads]] = PrivateAttr(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._compiled = self._compile()

    def _compile(self) -> list[tuple[str, Callable[..., Any], Reads]]:
        compiled = []
        keys: set[str] = set()
        for item in self.expressions:
            if not item.key.isidentifier():
                raise ValueError(f"Expression '{self.name}': key {item.key!r} is not a valid identifier")
            if item.key in keys:
                raise ValueError(f"Expression '{self.name}': key {item.key!r} is used twice")
            keys.add(item.key)
            try:
                # Compiled before its reads are collected, from the text wrapped in braces, so a syntax error
                # names the text as the author wrote it.
                expression = _ENVIRONMENT.compile_expression(item.expression, undefined_to_none=True)
                reads = read_paths(item.expression)
            except TemplateSyntaxError as e:
                raise ValueError(f"Expression '{self.name}': {item.key!r} is not a valid expression: {e}") from e
            expression, reads = refuse_clash(expression, reads, f"Expression '{self.name}': {item.key!r}")
            refuse_reserved_read(reads, f"Expression '{self.name}': {item.key!r}")
            compiled.append((item.key, expression, reads))
        return compiled

    def execute(self, input_data: ExpressionInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """Evaluates every expression against the inputs."""
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **{**kwargs, "parent_run_id": kwargs.get("run_id", uuid4())})

        context = input_data.model_dump()
        # The context goes in positionally: spread as keywords, an input named `self` would collide with the
        # compiled expression's own bound argument and fail the run before anything is evaluated. Such an input
        # passes through but is never read, since Jinja binds the name inside the expression and a read of it is
        # refused at build. An input named like a helper is visible where the expression reads it and hidden
        # where the expression calls the helper.
        # A missing input inside a list or a dict the expression builds is None there too.
        computed = {
            key: concrete(expression(scope_for(reads, context, _ENVIRONMENT.undefined)))
            for key, expression, reads in self._compiled
        }
        return {**context, **computed} if self.pass_through else computed
