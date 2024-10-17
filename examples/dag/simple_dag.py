import logging
import os

from dynamiq import Workflow, prompts
from dynamiq.connections import connections
from dynamiq.nodes.llms import Anthropic, OpenAI
from dynamiq.nodes.operators import Choice, ChoiceCondition, ChoiceOption, ConditionOperator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys and connections
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# No need to specify connection ID explicitly
openai_connection = connections.OpenAI(
    api_key=OPENAI_API_KEY,
)
anthropic_connection = connections.Anthropic(
    api_key=ANTHROPIC_API_KEY,
)

with Workflow() as wf:

    # OpenAI node 1
    openai_node = OpenAI(
        name="openai",
        model="gpt-3.5-turbo",
        connection=openai_connection,
        prompt=prompts.Prompt(
            messages=[
                prompts.Message(
                    role="user",
                    content="What is AI?",
                ),
            ],
        ),
    )
    wf.flow.add_nodes(openai_node)

    # OpenAI node 2 depending on OpenAI node 1
    openai_2_node = OpenAI(
        name="openai-2",
        model="gpt-3.5-turbo",
        connection=openai_connection,
        prompt=prompts.Prompt(
            messages=[
                prompts.Message(
                    role="user",
                    content="What is Data Science?",
                ),
            ],
        ),
    ).depends_on(openai_node)
    wf.flow.add_nodes(openai_2_node)

    # Anthropic node 1
    anthropic_node = Anthropic(
        name="anthropic",
        model="claude-3-opus-20240229",
        connection=anthropic_connection,
        prompt=prompts.Prompt(
            messages=[
                prompts.Message(
                    role="user",
                    content="What is an LLM?",
                ),
            ],
        ),
    )
    wf.flow.add_nodes(anthropic_node)

    # Choice node depending on OpenAI node 1
    choice_node = Choice(
        name="Choice",
        options=[
            ChoiceOption(
                id="choice_is_correct_date",
                name="choice_is_correct_date",
                condition=ChoiceCondition(
                    operator=ConditionOperator.STRING_EQUALS,
                    variable="$.date",
                    value="4 May 2024",
                ),
            ),
            ChoiceOption(
                id="choice_is_correct_next_date",
                name="choice_is_correct_next_date",
                condition=ChoiceCondition(
                    operator=ConditionOperator.STRING_EQUALS,
                    variable="$.next_date",
                    value="5 May 2024",
                ),
            ),
        ],
    ).depends_on(openai_node)
    wf.flow.add_nodes(choice_node)

    # Anthropic node 2, depending on multiple nodes including Choice
    anthropic_2_node = (
        Anthropic(
            name="anthropic-2",
            model="claude-3-opus-20240229",
            connection=anthropic_connection,
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
        )
        .enable_streaming()
        .depends_on([openai_node, openai_2_node, anthropic_node, choice_node])
    )
    wf.flow.add_nodes(anthropic_2_node)

    # OpenAI node 3 depending on OpenAI node 2 and choice node's next date option
    openai_3_node = OpenAI(
        name="openai-3",
        model="gpt-3.5-turbo",
        connection=openai_connection,
        prompt=prompts.Prompt(
            messages=[
                prompts.Message(
                    role="user",
                    content="What is RAG?",
                ),
            ],
        ),
    ).depends_on([openai_2_node, choice_node])
    wf.flow.add_nodes(openai_3_node)

    # OpenAI node 4 depending on OpenAI node 3
    openai_4_node = OpenAI(
        name="openai-4",
        model="gpt-3.5-turbo",
        connection=openai_connection,
        prompt=prompts.Prompt(
            messages=[
                prompts.Message(
                    role="user",
                    content="What is LLM fine-tuning?",
                ),
            ],
        ),
    ).depends_on(openai_3_node)
    wf.flow.add_nodes(openai_4_node)

    # OpenAI node 5 depending on OpenAI node 3 and OpenAI node 4 and used custom inputs and transformations
    def merge_and_short_content(inputs: dict, outputs: dict[str, dict]):
        return f"- {outputs[openai_4_node.id]['content'][:200]} \n - {outputs[openai_4_node.id]['content'][:200]}"

    openai_5_node = (
        OpenAI(
            name="openai-5",
            model="gpt-3.5-turbo",
            connection=openai_connection,
            prompt=prompts.Prompt(
                messages=[
                    prompts.Message(
                        role="user",
                        content=(
                            "Please simplify that information for {{purpose}}:\n"
                            "{{extra_instructions}}\n"
                            "{{content}}\n"
                            "{{extra_content}}"
                        ),
                    )
                ],
            ),
        )
        .inputs(
            purpose="10 years old kids",
            extra_instructions="Please return information in readable format.",
            content=merge_and_short_content,
            extra_content=openai_2_node.outputs.content,
        )
        .depends_on([openai_2_node, openai_3_node, openai_4_node])
    )
    wf.flow.add_nodes(openai_5_node)

    wf.run(input_data={"date": "4 May 2024", "next_date": "6 May 2024"})

    logger.info(f"Workflow {wf.id} finished. Results: ")

    for node_id, result in wf.flow._results.items():
        logger.info(f"Node {node_id} - {wf.flow._node_by_id[node_id].name}: {result}")
