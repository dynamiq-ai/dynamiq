import logging
import os
import uuid

from dynamiq import Workflow, prompts, runnables
from dynamiq.connections import connections
from dynamiq.connections.managers import ConnectionManager
from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer
from dynamiq.nodes.llms import Anthropic, OpenAI
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.operators import Choice, ChoiceCondition, ChoiceOption, ConditionOperator
from dynamiq.types.streaming import StreamingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


CM = ConnectionManager()


OPENAI_CONNECTION = connections.OpenAI(
    id=str(uuid.uuid4()),
    api_key=OPENAI_API_KEY,
)
ANTHROPIC_CONNECTION = connections.Anthropic(
    id=str(uuid.uuid4()),
    api_key=ANTHROPIC_API_KEY,
)
OPENAI_NODE = OpenAI(
    name="OpenAI",
    model="gpt-3.5-turbo",
    connection=OPENAI_CONNECTION,
    prompt=prompts.Prompt(
        messages=[
            prompts.Message(
                role="user",
                content="What is AI?",
            ),
        ],
    ),
    is_postponed_component_init=True,
)
OPENAI_2_NODE = OpenAI(
    name="OpenAI2",
    model="gpt-3.5-turbo",
    connection=OPENAI_CONNECTION,
    prompt=prompts.Prompt(
        messages=[
            prompts.Message(
                role="user",
                content="What is Data Science?",
            ),
        ],
    ),
    depends=[NodeDependency(OPENAI_NODE)],
    is_postponed_component_init=True,
)
ANTHROPIC_NODE = Anthropic(
    name="Anthropic",
    model="claude-3-opus-20240229",
    connection=ANTHROPIC_CONNECTION,
    prompt=prompts.Prompt(
        messages=[
            prompts.Message(
                role="user",
                content="What is LLM?",
            ),
        ],
    ),
    is_postponed_component_init=True,
)
CHOICE_NODE = Choice(
    name="Choice",
    options=[
        ChoiceOption(
            id="choice_is_correct_date",
            name="choice_is_correct_date",
            condition=ChoiceCondition(
                operator=ConditionOperator.STRING_EQUALS,
                variable="$.date",
                value="4 May 2024",
                operands=[],
            ),
        ),
        ChoiceOption(
            id="choice_is_correct_next_date",
            name="choice_is_correct_next_date",
            condition=ChoiceCondition(
                operator=ConditionOperator.STRING_EQUALS,
                variable="$.next_date",
                value="5 May 2024",
                operands=[],
            ),
        ),
    ],
    depends=[
        NodeDependency(OPENAI_NODE),
    ],
    is_postponed_component_init=True,
)

ANTHROPIC_2_NODE = Anthropic(
    name="Anthropic2",
    model="claude-3-opus-20240229",
    connection=ANTHROPIC_CONNECTION,
    input_transformer=InputTransformer(
        selector={
            "ai": f"$.{[OPENAI_NODE.id]}.content",
            "ds": f"$.{[OPENAI_2_NODE.id]}.content",
            "llm": f"$.{[ANTHROPIC_NODE.id]}.content",
        },
    ),
    prompt=prompts.Prompt(
        messages=[
            prompts.Message(
                role="user",
                content=(
                    "Please simplify that information for 10 years kids:\n"
                    "- {{ds}}\n\n"
                    "- {{llm}}\n\n"
                    "- {{ai}}\n\n"
                ),
            ),
        ],
    ),
    streaming=StreamingConfig(enabled=True),
    depends=[
        NodeDependency(OPENAI_NODE),
        NodeDependency(OPENAI_2_NODE),
        NodeDependency(ANTHROPIC_NODE),
        NodeDependency(CHOICE_NODE, option="choice_is_correct_date"),
    ],
    is_postponed_component_init=True,
)
OPENAI_3_NODE = OpenAI(
    name="OpenAI3",
    model="gpt-3.5-turbo",
    connection=OPENAI_CONNECTION,
    prompt=prompts.Prompt(
        messages=[
            prompts.Message(
                role="user",
                content="What is RAG?",
            ),
        ],
    ),
    depends=[
        NodeDependency(OPENAI_2_NODE),
        NodeDependency(CHOICE_NODE, option="choice_is_correct_next_date"),
    ],
    is_postponed_component_init=True,
)
OPENAI_4_NODE = OpenAI(
    name="OpenAI4",
    model="gpt-3.5-turbo",
    connection=OPENAI_CONNECTION,
    prompt=prompts.Prompt(
        messages=[
            prompts.Message(
                role="user",
                content="What is Fine-tuning?",
            ),
        ],
    ),
    depends=[
        NodeDependency(OPENAI_3_NODE),
    ],
    is_postponed_component_init=True,
)

NODES = [
    OPENAI_NODE,
    OPENAI_2_NODE,
    OPENAI_3_NODE,
    OPENAI_4_NODE,
    ANTHROPIC_NODE,
    ANTHROPIC_2_NODE,
    CHOICE_NODE,
]
WF = Workflow(id="wf", flow=Flow(id="wf", nodes=NODES, connection_manager=CM))


if __name__ == "__main__":
    WF.run(
        input_data={"date": "4 May 2024", "next_date": "6 May 2024"},
        config=runnables.RunnableConfig(callbacks=[]),
    )
    logger.info(f"Workflow {WF.id} finished. Results: ")
    for node_id, result in WF.flow._results.items():
        logger.info(f"Node {node_id}-{WF.flow._node_by_id[node_id].name}: {result}")
