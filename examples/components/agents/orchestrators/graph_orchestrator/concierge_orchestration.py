import re
from typing import Any

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.orchestrators.graph import END, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.tools.function_tool import function_tool
from dynamiq.nodes.tools.human_feedback import HumanFeedbackAction, HumanFeedbackTool
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm


def create_workflow() -> Workflow:
    """
    Create the workflow with all necessary agents and tools.

    Returns:
        Workflow: The configured workflow.
    """

    llm = setup_llm()

    # Stock Lookup Agent
    def stock_lookup(context: dict[str, Any], **kwargs):

        def search_for_stock_symbol(str: str) -> str:
            """Useful for searching for a stock symbol given a company name."""
            logger.info("Searching for stock symbol")
            return str.upper()

        @function_tool
        def lookup_stock_price(company_name: str, **kwargs) -> str:
            """Useful for looking up a stock price ."""

            stock_symbol = search_for_stock_symbol(company_name)
            logger.info(f"Looking up stock price for {stock_symbol}")

            context["current_task"] = ""
            context["task_result"] = f"{stock_symbol} is currently trading at $100.00"

            return f"Symbol {stock_symbol} is currently trading at $100.00"

        stock_lookup_agent = Agent(
            name="stock_lookup_agent",
            role="""You are a helpful assistant that is searching for stock prices.""",
            goal="Provide actions according to role",  # noqa: E501
            llm=llm,
            tools=[lookup_stock_price()],
        )

        company_name = input("Provide company name for which you want to find stock price.\n")
        result = stock_lookup_agent.run(
            input_data={"input": f"Get stock price for {company_name}."},
        )

        return {"result": result.output.get("content"), **context}

    # Auth Agent
    def authenticate(context: dict[str, Any], **kwargs):

        @function_tool
        def login(password: str, **kwargs) -> None:
            """Given a password, logs in and stores a session in the user state."""
            logger.info(f"Logging in with password {password}")
            context["authenticated"] = True
            return f"Sucessfully logged in with password {password}"

        @function_tool
        def is_authenticated(**kwargs) -> bool:
            """Checks if the user has saved session in user state."""
            logger.info("Checking if authenticated")
            return context.get("authenticated", False)

        auth_agent = Agent(
            name="auth_agent",
            role="""You are a helpful assistant that is authenticating a user.""",
            goal="Provide actions according to role",  # noqa: E501
            llm=llm,
            tools=[login(), is_authenticated()],
        )

        result = auth_agent.run(
            input_data={"input": ""},
        )

        return {"result": result.output.get("content"), **context}

    def account_balance(context: dict[str, Any], **kwargs):
        @function_tool
        def get_account_credentials(**kwargs) -> str:
            """Useful for looking up account ID."""
            logger.info("Searching the account ID.")
            return "Account ID - 1234567890"

        @function_tool
        def get_account_balance(account_id: str, **kwargs) -> str:
            """Useful for looking up an account balance. Account ID is required."""

            logger.info(f"Looking up account balance for {account_id}")
            context["current_task"] = ""
            context["task_result"] = "Account has a balance of $1000"

            return f"Account {account_id} has a balance of $1000"

        account_balance_agent = Agent(
            name="account_balance_agent",
            role="""
            You are a helpful assistant that is looking up account balances.
            """,
            goal="Provide actions according to role",  # noqa: E501
            llm=llm,
            tools=[get_account_credentials(), get_account_balance()],
        )

        result = account_balance_agent.run(
            input_data={"input": "Get account balance."},
        )

        return {"result": result.output.get("content"), **context}

    # Transfer Money Agent
    def transfer_money(context: dict[str, Any], **kwargs):
        @function_tool
        def transfer_money(from_account_id: str, to_account_id: str, amount: int, **kwargs) -> None:
            """Useful for transferring money between accounts."""

            logger.info(f"Transferring {amount} from {from_account_id} account {to_account_id}")

            context["current_task"] = ""
            context["task_result"] = f"{amount}$ was successfully transferred"

            return f"Transferred {amount} to account {to_account_id}"

        @function_tool
        def check_balance(account_id: str, amount: int, **kwargs) -> bool:
            """Useful for checking if an account has enough money.
            Account ID and amount of money to transfer are required."""

            logger.info(f"Checking if there is more than {amount} on account with ID - {account_id}")
            return "Balance has sufficient amount of money."

        transfer_money_agent = Agent(
            name="transfer_money_agent",
            role="""
            You are a helpful assistant that transfers money between accounts.
            """,
            goal="Provide actions according to role",  # noqa: E501
            llm=llm,
            tools=[transfer_money(), check_balance()],
        )

        sender_id = input("Provide account ID of sender.\n")
        receiver_id = input("Provide account ID of receiver.\n")
        amount = input("Provide amount of money you want to transfer.\n")

        result = transfer_money_agent.run(
            input_data={"input": f"Transfer {amount}$ from account {sender_id} to {receiver_id}."},
        )

        return {"result": result.output.get("content"), **context}

    human_feedback_tool = HumanFeedbackTool(
        action=HumanFeedbackAction.ASK,  # Always wait for user input
    )

    def concierge(context: dict[str, Any], **kwargs):
        if current_task := context.get("current_task"):
            return {"result": f"Proceed with task {current_task}"}

        else:

            if task_result := context.get("task_result"):
                input_text = f"{task_result}." " Anything else I can help with?"
            else:
                input_text = (
                    "Welcome to financial system! How do you want to continue:\n"
                    "* looking up a stock price\n"
                    "* authenticating the user\n"
                    "* checking an account balance (requires authentication first)\n"
                    "* transferring money between accounts (requires authentication)\n"
                )

            result = human_feedback_tool.run(
                input_data={"input": input_text},
            )

            output = result.output.get("content")
            context["current_task"] = output

            return {"result": output, **context}

    llm = llm
    agent_manager = GraphAgentManager(llm=llm)

    graph_orchestrator = GraphOrchestrator(manager=agent_manager, final_summarizer=True, initial_state="concierge")

    # Orchestration path function
    def orchestrate(context: dict[str, Any], **kwargs):

        formatted_prompt = f"""
            You are an orchestration agent.
            Your task is to decide and trigger the next state based on the user's current state and request.
            Only one state should be called.

            Just return name of the state:
            * stock_lookup - To find stock price
            * transfer_money - To transfer money
            * authenticate - To authentificate
            * account_balance - To check account balance
            * END - Finish execution

            {"Current task: " + context.get("current_task", "")}

            Conversation history:
            {context.get("history")}
            """
        result = llm.run(
            input_data={"input": "Assist user."},
            prompt=Prompt(messages=[Message(role="user", content=formatted_prompt)]),
        )

        pattern = r"\b(stock_lookup|transfer_money|authenticate|account_balance|END)\b"
        match = re.search(pattern, result.output["content"])

        if match:
            next_state = match.group()
            if next_state == "account_balance" and not context.get("authenticated"):
                return "authenticate"
            if next_state == "transfer_money" and not context.get("authenticated"):
                return "authenticate"
            return next_state

    # Create states
    graph_orchestrator.add_state_by_tasks("concierge", [concierge])
    graph_orchestrator.add_state_by_tasks("stock_lookup", [stock_lookup])
    graph_orchestrator.add_state_by_tasks("transfer_money", [transfer_money])
    graph_orchestrator.add_state_by_tasks("authenticate", [authenticate])
    graph_orchestrator.add_state_by_tasks("account_balance", [account_balance])

    # Add path to other nodes through concierge
    graph_orchestrator.add_conditional_edge(
        "concierge", ["stock_lookup", "transfer_money", "authenticate", "account_balance", END], orchestrate
    )

    # Add path back from specialized states to concierge
    graph_orchestrator.add_edge("stock_lookup", "concierge")
    graph_orchestrator.add_edge("authenticate", "concierge")

    # This states require authorization, orchestrator will not allow to get here without being authorized.
    graph_orchestrator.add_edge("account_balance", "concierge")
    graph_orchestrator.add_edge("transfer_money", "concierge")

    return Workflow(
        flow=Flow(nodes=[graph_orchestrator]),
    )


def run_workflow() -> tuple[str, dict]:
    """Runs workflow"""

    # Create workflow
    workflow = create_workflow()

    user_prompt = """
    Hello
    """  # noqa: E501

    # Run workflow
    tracing = TracingCallbackHandler()
    try:
        result = workflow.run(
            input_data={"input": user_prompt},
            config=RunnableConfig(callbacks=[tracing]),
        )

        logger.info("Workflow completed successfully")

        # Print and save result
        output = result.output[workflow.flow.nodes[0].id]["output"]["content"]

        return output, tracing.runs

    except Exception as e:
        logger.error(f"An error occurred during workflow execution: {str(e)}")
        return "", {}


if __name__ == "__main__":
    output, _ = run_workflow()
    print(output)
