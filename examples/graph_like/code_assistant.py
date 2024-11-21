import json
from typing import Any

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import E2B
from dynamiq.flows import Flow
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from examples.llm_setup import setup_llm


def create_orchestrator() -> GraphOrchestrator:
    """
    Creates orchestrator

    Returns:
        GraphOrchestrator: The configured orchestrator.
    """
    llm = setup_llm()
    connection_e2b = E2B()

    tool_code = E2BInterpreterTool(connection=connection_e2b)
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0)

    def code_llm(messages, structured_output=True):
        code_sample = {
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "name": "generate_code_solution",
                "schema": {
                    "type": "object",
                    "required": ["libraries", "code"],
                    "properties": {
                        "libraries": {
                            "type": "string",
                            "description": (
                                "Libraries that have to be installed (coma separated)." " Example: 'pandas,numpy'"
                            ),
                        },
                        "code": {"type": "string", "description": "Code solution to the problem."},
                    },
                    "additionalProperties": False,
                },
            },
        }

        llm_result = llm.run(
            input_data={},
            prompt=Prompt(
                messages=messages,
            ),
            schema=code_sample if structured_output else None,
            inference_mode=InferenceMode.STRUCTURED_OUTPUT if structured_output else InferenceMode.XML,
        ).output["content"]

        return json.loads(llm_result) if structured_output else llm_result

    def generate_code_solution(context: dict[str, Any]):
        """
        Generate a code solution
        """

        print("#####CODE GENERATION#####")

        messages = context.get("messages")

        if context.get("reiterate", False):
            messages += [Message(role="user", content="Generate code again taking into account errors. {}")]

        code_solution = code_llm(messages)
        context["solution"] = code_solution

        context["messages"] += [
            Message(
                role="assistant",
                content=f"\n Imports: {code_solution.get('libraries')} \n Code: {code_solution.get('code')}",
            )
        ]

        context["iterations_num"] += 1

        return {"result": code_solution, **context}

    def reflect(context: dict[str, Any]):
        print("#####REFLECTING ON ERRORS#####")
        reflections = code_llm(messages=context.get("messages"))
        context["messages"] += [Message(role="assistant", content=f"Here are reflections on the error: {reflections}")]
        return {"result": reflections, **context}

    def validate_code(context: dict[str, Any], **_):
        """
        Check code

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """

        print("#####CHECKING#####")
        solution = context["solution"]

        result = tool_code.run(input_data={"python": solution.get("code"), "packages": solution.get("libraries")})
        if result.status == RunnableStatus.SUCCESS:
            print("#####SUCCESSFUL#####")
            successful_message = [
                Message(role="user", content=f"Your code executed successfully {result.output['content']}")
            ]
            context["messages"] += successful_message
            context["reiterate"] = False
        else:
            print("#####FAILED#####")
            error_message = [
                Message(
                    role="user",
                    content=(
                        f"Your solution failed the code execution test: {result.output['content']}."
                        " Reflect on possible errors."
                    ),
                )
            ]
            context["messages"] += error_message
            context["reiterate"] = True

        return {"result": result.output.get("content"), **context}

    orchestrator = GraphOrchestrator(
        name="Graph orchestrator",
        manager=GraphAgentManager(llm=llm),
    )

    orchestrator.add_node("generate_code", [generate_code_solution])
    orchestrator.add_node("validate_code", [validate_code])
    orchestrator.add_node("reflect", [reflect])

    orchestrator.add_edge(START, "generate_code")
    orchestrator.add_edge("generate_code", "validate_code")
    orchestrator.add_edge("reflect", "generate_code")

    def orchestrate(context: dict[str, Any]) -> str:
        return "reflect" if context["reiterate"] else END

    orchestrator.add_conditional_edge("validate_code", ["generate_code", END], orchestrate)
    return orchestrator


def run_orchestrator(request="Write 100 lines of code.") -> RunnableResult:
    """Runs orchestrator"""
    orchestrator = create_orchestrator()
    orchestrator.context = {
        "messages": [Message(role="user", content=request)],
        "iterations_num": 0,
        "reiterate": False,
    }

    tracing = TracingCallbackHandler()

    workflow = Workflow(flow=Flow(nodes=[orchestrator]))

    result = workflow.run(
        input_data={"input": "Provide final code that succeed and reflection on coding process."},
        config=RunnableConfig(callbacks=[tracing]),
    )

    print(result.output[orchestrator.id])
    return result.output[orchestrator.id]["output"]["content"], tracing.runs


if __name__ == "__main__":
    result, _ = run_orchestrator()
    print("Result:")
    print(result)
