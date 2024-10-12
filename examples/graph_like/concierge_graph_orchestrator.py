import json

from dotenv import load_dotenv

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.orchestrators.graph import GraphOrchestrator, START, END, BaseContext

from dynamiq.nodes.agents.react import ReActAgent

from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm
from dynamiq.nodes.tools.function_tool import function_tool
from dynamiq.nodes.tools.human_feedback import HumanFeedbackTool
from dynamiq.nodes.agents.base import Agent
from dynamiq.prompts import Message, Prompt
import re

class CustomToolAgent(Agent):
    def execute(
        self, input_data, config: RunnableConfig | None = None, **kwargs
    ):
        """
        Executes the agent with the given input data.
        """
        logger.debug(f"Agent {self.name} - {self.id}: started with input {input_data}")
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        tool_result = self.tools[0].run(
            input_data=input_data,
            config=config,
            **kwargs,
        )

        execution_result = {
            "content": tool_result,
            "intermediate_steps": self._intermediate_steps,
        }

        if self.streaming.enabled:
            self.run_on_node_execute_stream(
                config.callbacks, execution_result, **kwargs
            )

        logger.debug(f"Agent {self.name} - {self.id}: finished with result {tool_result}")
        return execution_result.output.get('context')


# Load environment variables
load_dotenv()

class ConciergeContext(BaseContext):
    authenticated: bool = False
    current_task: str = ''


def create_workflow() -> Workflow:
    """
    Create the workflow with all necessary agents and tools.

    Returns:
        Workflow: The configured workflow.
    """

    llm = setup_llm()

    # Stock Lookup Agent

    def stock_lookup(ctx: ConciergeContext, **kwargs):
        @function_tool
        def lookup_stock_price(stock_symbol: str) -> str:
            """Useful for looking up a stock price."""
            logger.info(f"Looking up stock price for {stock_symbol}")
            ctx.current_task = ''
            return f"Symbol {stock_symbol} is currently trading at $100.00"

        @function_tool
        def search_for_stock_symbol(str: str) -> str:
            """Useful for searching for a stock symbol given a free-form company name."""
            logger.info("Searching for stock symbol")
            return str.upper()


        stock_lookup_agent = ReActAgent(
            name="stock_lookup_agent",
            role="""You are a helpful assistant that is looking up stock prices.
            The user may not know the stock symbol of the company they're interested in,
            so you can help them look it up by the name of the company.
            You can only look up stock symbols given to you by the search_for_stock_symbol tool, don't make them up.
            Trust the output of the search_for_stock_symbol tool even if it doesn't make sense to you.""",
            goal="Provide actions according to role",  # noqa: E501
            llm=llm,
            tools=[lookup_stock_price(), search_for_stock_symbol()],
        )
        result = stock_lookup_agent.run(
                input_data={
                    "input": 'Get stock price for nvidia.'
                },
            )

        output = result.output.get('content')
        return output

    # Auth Agent

    def authentificate(context: ConciergeContext, config: RunnableConfig = None):
        @function_tool
        def store_username(username: str) -> None:
            """Adds the username to the user state."""
            print("Recording username")
            return "Username was recorded."

        @function_tool
        def login(password: str) -> None:
            """Given a password, logs in and stores a session token in the user state."""
            logger.info(f"Logging in with password {password}")
            context.authenticated = True
            return f"Sucessfully logged in with password {password}"

        @function_tool
        def is_authenticated() -> bool:
            """Checks if the user has a session token."""
            print("Checking if authenticated")
            return "User is authenticated"

        @function_tool
        def done() -> None:
            """When you complete your task, call this tool."""
            logger.info("Authentication is complete")
            return "Authentication is complete"

        auth_agent = ReActAgent(
            name="auth_agent",
            role="""You are a helpful assistant that is authenticating a user.
            Your task is to get a valid session token stored in the user state.
            To do this, the user must supply you with a username and a valid password. You can ask them to supply these.
            If the user supplies a username and password, call the tool "login" to log them in.""",
            goal="Provide actions according to role",  # noqa: E501
            llm=llm,
            tools=[store_username(), login(), is_authenticated(), done()],
            )
        result = auth_agent.run(
            input_data={
                "input": ""
            },
            config=config,
        )
        
        return result
        
    def account_balance(ctx: ConciergeContext, **kwargs):
    # Account Balance Agent
        @function_tool
        def get_account_id(account_name: str) -> str:
            """Useful for looking up an account ID."""
            print(f"Looking up account ID for {account_name}")
            account_id = "1234567890"
            return f"Account id is {account_id}"

        @function_tool
        def get_account_name() -> str:
            """Useful for looking up an account name."""
            print(f"Looking up account for account name")
            account_name = "john123"
            return f"Account name is {account_name}"

        @function_tool
        def get_account_balance(account_id: str) -> str:
            """Useful for looking up an account balance."""
            logger.info(f"Looking up account balance for {account_id}")
            ctx.current_task = ''
            return f"Account {account_id} has a balance of $1000"

        @function_tool
        def is_authenticated() -> bool:
            """Checks if the user has a session token."""
            logger.info("Account balance agent is checking if authenticated")
            return "User is authentificated"


        account_balance_agent = ReActAgent(
            name="account_balance_agent",
            role="""
            You are a helpful assistant that is looking up account balances.
            The user may not know the account ID of the account they're interested in,
            so you can help them look it up by the name of the account.
            If they're trying to transfer money, they have to check their account balance first, which you can help with.
            """,
            goal="Provide actions according to role",  # noqa: E501
            llm=llm,
            tools=[get_account_id(), get_account_balance(), is_authenticated(), get_account_name()],
        )

        result = account_balance_agent.run(
                input_data={
                    "input": 'Get account balance.'
                },
            )

        output = result.output.get('content')
        return output

    # Transfer Money Agent

    def transfer_money(ctx: ConciergeContext, **kwargs):
        @function_tool
        def transfer_money(from_account_id: str, to_account_id: str, amount: int) -> None:
            """Useful for transferring money between accounts."""
            logger.info(f"Transferring {amount} from {from_account_id} account {to_account_id}")
            ctx.current_task = ''
            return f"Transferred {amount} to account {to_account_id}"

        @function_tool
        def balance_sufficient(account_id: str, amount: int) -> bool:
            """Useful for checking if an account has enough money to transfer."""
            logger.info("Checking if balance is sufficient")
            return "There is enough money."

        @function_tool
        def has_balance() -> bool:
            """Useful for checking if an account has a balance."""
            logger.info("Checking if account has a balance")
            return "Account has enough balance"

        @function_tool
        def is_authenticated() -> bool:
            """Checks if the user has a session token."""
            logger.info("Transfer money agent is checking if authenticated")
            return "User has a session token."

        transfer_money_agent = ReActAgent(
            name="transfer_money_agent",
            role="""
            You are a helpful assistant that transfers money between accounts.
            The user can only do this if they are authenticated, which you can check with the is_authenticated tool.
            If they aren't authenticated, tell them to authenticate first.
            The user must also have looked up their account balance already, which you can check with the has_balance tool.
            """,
            goal="Provide actions according to role",  # noqa: E501
            llm=llm,
            tools=[transfer_money(), balance_sufficient(), has_balance(), is_authenticated()],
        )
        result = transfer_money_agent.run(
                input_data={
                    "input": 'Transfer 100$ from account 132 to 123.'
                },
        )

        output = result.output.get('content')

        return output

    human_feedback_tool = HumanFeedbackTool()
    def concierge(ctx: ConciergeContext, **kwargs):
        if ctx.current_task:
            return f"Proceed with task {ctx.current_task}"

        else:
            result = human_feedback_tool.run(
                input_data={
                    "input": (
                "Welcome to financial system! How do you want to continue:"
                "* looking up a stock price"
                "* authenticating the user"
                "* checking an account balance (requires authentication first)"
                "* transferring money between accounts (requires authentication and checking an account balance first)")
                },
            )

            output = result.output.get('content')
            ctx.current_task = output
            return output
            

    llm = llm
    agent_manager = GraphAgentManager(llm=llm)

    graph_orchestrator = GraphOrchestrator(
        manager=agent_manager,
        final_summarizer=True,
        context = ConciergeContext(),
        initial_state = 'concierge'
    )


    # Orchestration path function
    def orchestration(context: ConciergeContext):
        
        formatted_prompt = f"""
            You are on orchestration agent.
            Your job is to decide which state to run based on the current state of the user and what they've asked to do. 
            You run an next state by calling the appropriate state name.
            You do not need to call more than one state.
            
            Just return name of the state you want to do next:
            * stock_lookup - To find stock price
            * transfer_money - To transfer money
            * authenticate - To authentificate
            * account_balance - To check account balance
            * END - Finish execution

            {"Current task: " + context.current_task if context.current_task else ""}

            Conversation history:
            {context.history}
            """
        result = llm.run(input_data={},
                prompt=Prompt(
                messages=[Message(role="user", content=formatted_prompt)]
                    ))

        pattern = r'\b(stock_lookup|transfer_money|authenticate|account_balance|END)\b'
        match = re.search(pattern, result.output['content'])

        if match:
            next_state = match.group()
            if next_state == 'account_balance' and not context.authenticated:
                return 'authenticate'
            if next_state == 'transfer_money' and not context.authenticated:
                return 'authenticate'
            return next_state

    # Create states
    graph_orchestrator.add_node('concierge', [concierge])
    graph_orchestrator.add_node('stock_lookup', [stock_lookup])
    graph_orchestrator.add_node('transfer_money', [transfer_money])
    graph_orchestrator.add_node('authenticate', [authentificate])
    graph_orchestrator.add_node('account_balance', [account_balance])

    # Add path to other nodes through concierge
    graph_orchestrator.add_conditional_edge('concierge', ['stock_lookup', 'transfer_money', 'authenticate', 'account_balance', END], orchestration)

    # Add path back from specialized states to concierge
    graph_orchestrator.add_edge('stock_lookup', 'concierge')
    graph_orchestrator.add_edge('authenticate', 'concierge')

    # This states require authorization, orchestrator will not allow to get here without being authorized. 
    graph_orchestrator.add_edge('account_balance', 'concierge')
    graph_orchestrator.add_edge('transfer_money', 'concierge')


    return Workflow(
        flow=Flow(nodes=[graph_orchestrator]),
    )


def run_planner() -> tuple[str, dict]:
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
        output = result.output[workflow.flow.nodes[0].id]['output']['content']

        return output, tracing.runs

    except Exception as e:
        logger.error(f"An error occurred during workflow execution: {str(e)}")
        return "", {}


if __name__ == "__main__":
    output, tracing = run_planner()
    print(output)