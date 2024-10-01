import logging
import os
import uuid

from dynamiq import Workflow, runnables
from dynamiq.connections import connections
from dynamiq.connections.managers import ConnectionManager
from dynamiq.flows import Flow
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.nodes.operators import Choice, ChoiceCondition, ChoiceOption, operators
from dynamiq.nodes.validators import ValidPython
from dynamiq.prompts import prompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CM = ConnectionManager()

OPENAI_CONNECTION = connections.OpenAI(
    id=str(uuid.uuid4()),
    api_key=OPENAI_API_KEY,
)

OPENAI_NODE = OpenAI(
    name="OpenAI",
    model="gpt-3.5-turbo",
    connection=OPENAI_CONNECTION,
    prompt=prompts.Prompt(
        messages=[
            prompts.Message(
                role="user",
                content="Give me please valid python code string,without additional messages and info",
            ),
        ],
    ),
    is_postponed_component_init=True,
)

VALID_PYTHON_NODE = ValidPython(
    depends=[
        NodeDependency(OPENAI_NODE),
    ],
    input_transformer=InputTransformer(
        selector={
            "content": f"${[OPENAI_NODE.id]}.output.content",
        },
    ),
)

CHOICE_NODE = Choice(
    name="Choice",
    input_transformer=InputTransformer(
        selector={
            "value": f"${[VALID_PYTHON_NODE.id]}.output.valid",
        },
    ),
    options=[
        ChoiceOption(
            id="choice_is_valid_code",
            name="choice_is_valid_code",
            condition=ChoiceCondition(
                operator=operators.ConditionOperator.BOOLEAN_EQUALS,
                variable="value",
                value=True,
                operands=[],
            ),
        ),
    ],
    depends=[
        NodeDependency(VALID_PYTHON_NODE),
    ],
    is_postponed_component_init=True,
)

WF = Workflow(
    id="wf",
    flow=Flow(
        id="wf",
        nodes=[OPENAI_NODE, VALID_PYTHON_NODE, CHOICE_NODE],
        connection_manager=CM,
    ),
)

if __name__ == "__main__":
    WF.run(
        input_data={},
        config=runnables.RunnableConfig(callbacks=[]),
    )
    logger.info(f"Workflow {WF.id} finished. Results: ")
    for node_id, result in WF.flow._results.items():
        logger.info(f"Node {node_id}-{WF.flow._node_by_id[node_id].name}: {result}")
