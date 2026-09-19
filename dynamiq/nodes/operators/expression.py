from collections.abc import Callable
from typing import Any, ClassVar, Literal
from uuid import uuid4

from jinja2 import TemplateSyntaxError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import BaseModel, ConfigDict, PrivateAttr

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.types import ExpressionItem, NamedField
from dynamiq.runnables import RunnableConfig

# One sandbox for every Expression node: it keeps no state, and it refuses attribute access that would
# reach Python internals, so an expression cannot escape into the process.
_ENVIRONMENT = ImmutableSandboxedEnvironment()


class ExpressionInputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class Expression(Node):
    """Computes new fields from named inputs with Jinja2 expressions.

    Each expression is the body of a template variable without the braces (`price * quantity`,
    `'A' if score < 0.5 else 'B'`, `amount | round(2)`), compiled once in a sandboxed environment and
    evaluated with the inputs as its names. It returns the value itself, not rendered text. An input
    referred to on its own that is missing evaluates to None; using a missing input in arithmetic
    fails the run, as does an expression that reaches for Python internals. The output holds one key
    per expression, plus every input when `pass_through` is set, with expressions winning on a clash.
    """

    name: str | None = "expression"
    group: Literal[NodeGroup.OPERATORS] = NodeGroup.OPERATORS
    input_fields: list[NamedField] = []
    expressions: list[ExpressionItem] = []
    pass_through: bool = False
    input_schema: ClassVar[type[ExpressionInputSchema]] = ExpressionInputSchema

    _compiled: list[tuple[str, Callable[..., Any]]] = PrivateAttr(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._compiled = self._compile()

    def _compile(self) -> list[tuple[str, Callable[..., Any]]]:
        compiled = []
        keys: set[str] = set()
        for item in self.expressions:
            if not item.key.isidentifier():
                raise ValueError(f"Expression '{self.name}': key {item.key!r} is not a valid identifier")
            if item.key in keys:
                raise ValueError(f"Expression '{self.name}': key {item.key!r} is used twice")
            keys.add(item.key)
            try:
                compiled.append((item.key, _ENVIRONMENT.compile_expression(item.expression, undefined_to_none=True)))
            except TemplateSyntaxError as e:
                raise ValueError(f"Expression '{self.name}': {item.key!r} is not a valid expression: {e}") from e
        return compiled

    def execute(self, input_data: ExpressionInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """Evaluates every expression against the inputs."""
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **{**kwargs, "parent_run_id": kwargs.get("run_id", uuid4())})

        context = input_data.model_dump()
        computed = {key: expression(**context) for key, expression in self._compiled}
        return {**context, **computed} if self.pass_through else computed
