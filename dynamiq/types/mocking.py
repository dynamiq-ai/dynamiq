"""Mock execution for nodes — suppressing a node's side effect.

A mocked node is **never executed**: the framework short-circuits the node's
``execute`` call and returns a synthetic result in its place. Everything else
about the run is unchanged — dependency validation, input transformation,
input-schema validation, output transformation, callbacks and tracing all still
happen. The only thing that does not happen is the side effect.

.. rubric:: Two different things are called "dry run" in this codebase

Read this first; the names are close and the meanings are not.

===========================  ==========================================================
``RunnableConfig.dry_run``   RAG ingestion. The nodes **do run**; documents and
(:mod:`dynamiq.types.        collections written during the run are deleted afterwards.
dry_run`)                    About *reversibility*. Surfaced in the UI as the run
                             modal's "Dry run" toggle.
``Node.mock`` /              This module. The node **does not run** at all. About
``RunnableConfig.mock``      *suppression*. Surfaced in the UI as a per-node
(this module)                "Mock execution" section.
===========================  ==========================================================

They are independent and compose: a run can clean up its RAG writes *and* mock
its payment tool. The field is called ``mock`` rather than ``dry_run`` so that
``node.mock`` (suppression) is never confused with ``node.dry_run_cleanup()``
(the RAG hook every node already inherits from ``DryRunMixin``).

.. rubric:: Where the seam sits

Mocking is checked *below* validation and *above* execution, deliberately: a dry
run should still fail when an agent calls a tool with malformed arguments.

**Approval.** A mocked node does not request human-in-the-loop approval, even when
``approval.enabled`` is set. Approval authorizes a real side effect; a mocked node has
none, so the standard prompt ("Approve or cancel execution") would ask a human to
authorize something that cannot happen — and would block unattended dry runs on every
approval-gated tool. To rehearse an approval step, leave that node unmocked. Note that
``HumanFeedbackTool`` is itself an ordinary node, so mocking *it* is the supported way
to supply a canned human answer and run a human-in-the-loop workflow unattended.

**Scope.** The seam lives in ``Node.run_sync`` / ``Node._run_async_native``, which is
how flows and agents invoke nodes — so every node on a canvas and every tool an agent
calls is covered. It is not consulted by the handful of composite nodes that call a
sub-component's ``execute()`` directly (an LLM ranker invoking its own LLM, for
example); those are internal component calls rather than node runs, and they bypass
caching and per-node tracing for the same reason.

Two use cases drive the design:

* **Sensitive tools.** A tool that writes to production is configured with
  ``mock.enabled = True`` (optionally ``locked = True``) so it can sit on a
  canvas and take part in a run without ever firing.
* **A/B testing.** The same workflow is run twice — once for real, once with
  ``RunMockConfig(policy=MockPolicy.ALL)`` — to compare agent behaviour against
  a side-effect-free baseline.

.. rubric:: Output shape

A mock with no authored ``output`` produces ``{"content": ...}``, which is the
tool-output contract agents rely on. That is why ``MockPolicy.ALL`` defaults to
sweeping in tools only.

A ``dict`` output is returned **verbatim** — nothing is added, annotated or
renamed — so the author owns the whole shape. Mock a retriever with
``{"documents": [...]}`` and its downstream nodes see exactly that; mock a tool and
remember to include ``content``, because the agent reads that key. A mocked run is
still identifiable from ``metadata.is_mocked`` in the trace, which is set for every
mock regardless of the output shape.

.. rubric:: Known limits

*MCP tool schemas still come from discovery.* An ``MCPServer`` is a discovery
wrapper: the tool names and schemas an agent can call come from a live ``list_tools``
call at agent construction, and mocking suppresses execution, not discovery. When a
**pinned** server (``enabled`` and ``locked``) is unreachable the run degrades instead of
dying — a warning is logged and the agent simply does not see that server's tools this
run. Discovery happens before any ``RunnableConfig`` exists, so it cannot know whether a
given run will mock the server; only a pin is guaranteed to hold under every policy, so
only a pinned server's tools can never be needed for real. For full fidelity (the agent
sees and "calls" the real tool names), the server must be reachable at startup; only
execution is suppressed. An unreachable server that is *unmocked*, or mocked without
``locked``, still fails loudly — a run that turns that mock off with ``MockPolicy.NONE``
or an exclusion needs the real tools, and would otherwise proceed silently tool-less.

*Mocking is scoped per node and per run, not per subtree.* Two agents sharing one tool
instance cannot mock it differently: ``resolve_mock`` sees the node, not the caller. For
an A/B run where one agent acts and another does not, give each agent its own tool
instances and select them with ``exclude_ids``.

.. rubric:: Template rendering

A ``str`` output is rendered with Jinja, like ``approval.msg_template``,
``Prompt`` content and the text transformer. Templates are workflow-author input
and are **not** sandboxed anywhere in this codebase; treat authoring a workflow as
a trusted operation. The zero-config path renders no template at all.
"""

import copy
import reprlib
from enum import Enum
from typing import Any

from jinja2 import Template
from pydantic import BaseModel, Field, field_validator

DEFAULT_MOCK_MARKER = "[MOCKED]"

# The echoed call arguments land in the agent's prompt and in the trace, so they are bounded.
# A caller-supplied `output` template is not truncated — that payload is authored, not incidental.
ECHO_INPUT_MAX_CHARS = 1000

# Truncates *while* rendering, so a huge validated input is never fully materialized as text
# just to be sliced away. Same idiom as the payload logger in dynamiq.nodes.node.
_ECHO_REPR = reprlib.Repr()
_ECHO_REPR.maxstring = ECHO_INPUT_MAX_CHARS
_ECHO_REPR.maxother = ECHO_INPUT_MAX_CHARS
_ECHO_REPR.maxdict = 20
_ECHO_REPR.maxlist = 20
_ECHO_REPR.maxlevel = 4


class MockPolicy(str, Enum):
    """Run-level policy deciding *which* nodes are mocked for a given run.

    Attributes:
        NODE: Honour each node's own ``mock`` config. The default.
        ALL: Mock every node in ``RunMockConfig.groups`` regardless of its own
            config — a whole-workflow dry run.
        NONE: Execute everything for real, ignoring node-level ``mock`` configs.
            Mocks marked ``locked`` are still honoured.
    """

    NODE = "node"
    ALL = "all"
    NONE = "none"


class MockConfig(BaseModel):
    """Per-node mock configuration — suppress this node and return a synthetic result.

    Attributes:
        enabled (bool): Replace this node's execution with a synthetic result.
        locked (bool): Keep mocking even under ``MockPolicy.NONE``. Use for
            nodes that must never fire — production writes, payments, emails.
        output (str | dict | None): The synthetic result. A ``dict`` is
            returned as the node output verbatim. A ``str`` is Jinja-rendered
            and wrapped as ``{"content": ...}``. ``None`` synthesises a
            description of the suppressed call.
        latency_seconds (float): Artificial delay, to keep timing-sensitive
            behaviour comparable to a real run.
        marker (str | None): Prefix applied to string output so a mocked result
            is visually unmistakable in transcripts and traces. ``None``
            disables the prefix.
    """

    enabled: bool = False
    locked: bool = Field(
        default=False,
        description="Keep this mock active even when the run requests MockPolicy.NONE.",
    )
    output: str | dict[str, Any] | None = Field(
        default=None,
        description=(
            "Synthetic node output. dict is used verbatim; str is Jinja-rendered against "
            "the node input and wrapped as {'content': ...}; None synthesises a description "
            "of the suppressed call."
        ),
    )
    error: str | None = Field(
        default=None,
        description=(
            "Fail instead of returning `output`, with this message. Lets a dry run exercise an "
            "agent's error-handling path without a real failure. Takes precedence over `output`. "
            "Raised once, not once per attempt: the mock seam replaces `execute_with_retry`, so "
            "`error_handling.max_retries` is not applied to an injected failure the way it would "
            "be to a real one."
        ),
    )
    latency_seconds: float = Field(default=0.0, ge=0.0, description="Artificial delay before returning.")
    marker: str | None = Field(
        default=DEFAULT_MOCK_MARKER,
        description=(
            "Prefix applied to string output (and injected error messages) so a mocked result is "
            "visually unmistakable in transcripts and traces. None disables the prefix."
        ),
    )

    @property
    def is_pinned(self) -> bool:
        """Whether this mock survives a run-level request to execute for real.

        ``locked`` only means anything for a mock that is on; a locked-but-disabled
        config is inert rather than surprising.
        """
        return self.enabled and self.locked

    @property
    def has_authored_payload(self) -> bool:
        """Whether someone wrote a specific result for this node, rather than leaving it generic."""
        return self.output is not None or self.error is not None

    def to_dict(self, for_tracing: bool = False, **kwargs) -> dict:
        """Serialize the config, collapsing it to a flag when off and tracing."""
        if for_tracing and not self.enabled:
            return {"enabled": False}
        return self.model_dump(**kwargs)

    @staticmethod
    def describe_skipped_call(node_name: str, input_data: Any) -> str:
        """Describe the call that did not happen, for a mock with no authored output.

        Deliberately not a Jinja template: the wording is a constant and the arguments are
        data, so the zero-config path — the one ``MockPolicy.ALL`` uses for every node it
        sweeps in — never runs a template engine at all.
        """
        rendered_input = _ECHO_REPR.repr(input_data)
        if len(rendered_input) > ECHO_INPUT_MAX_CHARS:
            # reprlib bounds each element, not the total: maxdict x maxlist x maxstring multiply.
            # The hard cut is what actually holds the promise, exactly as the payload logger does.
            rendered_input = f"{rendered_input[:ECHO_INPUT_MAX_CHARS]}... [truncated]"
        return (
            f"Execution of '{node_name}' was skipped: this node is configured for mock "
            f"execution and produced no real side effect. "
            f"It was called with: {rendered_input}"
        )

    def render(self, node_name: str, node_id: str, input_data: Any) -> dict[str, Any]:
        """Build the synthetic node output for one mocked call.

        Args:
            node_name: Name of the node being mocked, for the echo template.
            node_id: Id of the node being mocked, exposed to the template.
            input_data: The validated input the node would have executed with.

        Returns:
            dict[str, Any]: The node output, always carrying a ``content`` key so
            it satisfies the tool-output contract agents rely on.
        """
        if isinstance(self.output, dict):
            # Verbatim: an authored dict owns the whole output shape, so a retriever mock can be
            # {"documents": [...]} without this adding keys its downstream nodes cannot read.
            # The run is still identifiable as mocked through `metadata.is_mocked` in the trace.
            # Deep-copied because one config serves every call: a Map over 1000 items, or an agent
            # calling the tool repeatedly, would otherwise hand out the same nested objects, and a
            # single downstream mutation would rewrite the mock for every later call.
            return copy.deepcopy(self.output)
        if self.output is None:
            output = {"content": self.describe_skipped_call(node_name, input_data)}
        else:
            output = {
                "content": Template(self.output).render(
                    marker=self.marker or "",
                    node_name=node_name,
                    node_id=node_id,
                    input=input_data,
                    input_data=input_data,
                )
            }

        if self.marker and isinstance(output.get("content"), str):
            content = output["content"]
            if not content.startswith(self.marker):
                output["content"] = f"{self.marker} {content}".strip()

        output["is_mocked"] = True
        return output


class RunMockConfig(BaseModel):
    """Run-level override for node mocking, set on ``RunnableConfig.mock``.

    Lets a single workflow definition be run for real, as a full dry run, or
    with its configured mocks suppressed — without editing any node.

    Attributes:
        policy (MockPolicy): Which nodes to mock. See :class:`MockPolicy`.
        groups (set[str]): Node groups swept in by ``MockPolicy.ALL``, matched
            against ``Node.group``. Defaults to tools only, so agents keep
            reasoning with real LLMs while their side effects are suppressed.
        exclude_ids (set[str]): Node ids kept live under every policy — so a read-only
            search tool can stay live during an otherwise complete dry run. A ``locked``
            mock still wins: a pin says the node must never fire, and an exclude list is
            not allowed to override that.
        exclude_names (set[str]): Same, matched by name. Names are not unique; see
            the field description before using it.
        default (MockConfig): Config used for nodes swept in by ``MockPolicy.ALL``
            that have no authored ``output`` or ``error`` of their own.
    """

    policy: MockPolicy = MockPolicy.NODE
    groups: set[str] = Field(
        default_factory=lambda: {"tools"},
        description="Node groups mocked under MockPolicy.ALL, matched against Node.group.",
    )
    exclude_ids: set[str] = Field(
        default_factory=set,
        description=(
            "Node ids kept live under every policy, except nodes whose mock is `locked` — a pin "
            "outranks an exclusion. Unambiguous — prefer this over `exclude_names`."
        ),
    )
    exclude_names: set[str] = Field(
        default_factory=set,
        description=(
            "Node names kept live, with the same `locked` carve-out as `exclude_ids`. Names are "
            "class defaults and are NOT unique — "
            "'api-call' matches every HttpApiCall — so a name here can un-mock more nodes than "
            "intended, including one deliberately mocked because it writes to production. Use "
            "`exclude_ids` unless you mean every node with the name."
        ),
    )
    default: MockConfig = Field(
        default_factory=lambda: MockConfig(enabled=True),
        description="Mock config applied to nodes swept in by MockPolicy.ALL that have none of their own.",
    )

    @field_validator("groups")
    @classmethod
    def validate_groups(cls, groups: set[str]) -> set[str]:
        """Reject unknown group names.

        Groups are plain strings rather than ``NodeGroup`` because ``dynamiq.nodes``
        imports this module, so the enum cannot be imported at module scope. Validating
        here keeps the ergonomics of strings without the failure mode: an unrecognised
        group would otherwise sweep in nothing and look like a mock that silently did
        not apply.
        """
        from dynamiq.nodes.types import NodeGroup

        known = {group.value for group in NodeGroup}
        if unknown := sorted(groups - known):
            raise ValueError(f"Unknown node group(s) {unknown}. Valid groups: {sorted(known)}.")
        return groups

    def resolve(
        self,
        node_mock: MockConfig,
        node_ids: set[str],
        node_names: set[str],
        node_group: str,
        sweepable: bool = True,
    ) -> MockConfig | None:
        """Decide the effective mock config for one node in this run.

        Args:
            node_mock: The mock config governing the node (see ``Node.mock_config``).
            node_ids: Ids this node can be matched by. More than one when the config
                came from an owner — an ``MCPTool`` also answers to its ``MCPServer``.
            node_names: Names this node can be matched by, same reasoning.
            node_group: The node's group value.
            sweepable: Whether ``MockPolicy.ALL`` may sweep this node in. False for the
                framework's own machinery, which an operator never means by "mock the
                tools"; an explicit node-level mock still applies.

        Returns:
            MockConfig | None: The config to mock with, or ``None`` to execute
            the node for real.
        """
        if (node_ids & self.exclude_ids) or (node_names & self.exclude_names):
            return node_mock if node_mock.is_pinned else None

        match self.policy:
            case MockPolicy.NONE:
                return node_mock if node_mock.is_pinned else None
            case MockPolicy.ALL:
                if node_mock.enabled:
                    return node_mock
                if not sweepable or node_group not in self.groups:
                    return None
                # An output someone authored but left switched off is still their intent for this
                # node; a whole-run dry run should use it rather than the generic echo.
                if node_mock.has_authored_payload:
                    return node_mock.model_copy(update={"enabled": True})
                return self.default if self.default.enabled else None
            case _:
                return node_mock if node_mock.enabled else None
