from bank_api import HOST, PORT

from dynamiq import Workflow
from dynamiq.connections import Http as HttpConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.embedders import OpenAITextEmbedder
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.retrievers import PineconeDocumentRetriever
from dynamiq.nodes.tools.http_api_call import HttpApiCall
from dynamiq.nodes.tools.human_feedback import HumanFeedbackTool
from dynamiq.nodes.tools.retriever import RetrievalTool


def run_workflow(input: str) -> str:
    # Create connection to OpenAI
    connection = OpenAIConnection()
    llm = OpenAI(
        connection=connection,
        model="gpt-4o-mini",
        temperature=0.01,
    )

    text_embedder = OpenAITextEmbedder(model="text-embedding-ada-002")
    document_retriever = PineconeDocumentRetriever(
        index_name="default",
        top_k=3,
    )

    bank_retriever_tool = RetrievalTool(
        name="Bank FAQ Search",
        text_embedder=text_embedder,
        document_retriever=document_retriever,
    )

    # Create a ReActAgent for handling bank documentation queries
    agent_bank_documentation = ReActAgent(
        name="RAG Agent",
        role="Customer support assistant for Internal Bank Documentation.",
        llm=llm,
        tools=[bank_retriever_tool],
    )

    # Create connection to Bank API
    connection = HttpConnection(
        method="POST",
        url=f"http://{HOST}:{PORT}/",
    )

    # Create api call tool
    api_call = HttpApiCall(
        connection=connection,
        name="Bank API",
        description="""
        An internal bank API.

        Available endpoints:
        * 'block_card' (int card_number, int pin_code)
        * 'make_transaction' (int card_number_sender, int card_number_reciever, int amount)
        * 'request_report' (int card_number, int pin_code)

        Choose between endpoints and pass name of it in url_path parameter.
        Parameters for endpoint have to be passed in `data` object.
        """,
    )

    # Create user interaction tool
    human_feedback_tool = HumanFeedbackTool()

    def combine_inputs(_: dict, outputs: dict[str, dict]):
        return f"Request: {input}\n" f"Follow this instruction: {outputs[agent_bank_documentation.id]['content']}"

    # Create a ReActAgent for handling internal bank API queries
    agent_bank_support = ReActAgent(
        name="API Agent",
        role="Customer support assistant with access to Internal Bank API",
        llm=llm,
        tools=[api_call, human_feedback_tool],
        depends=[NodeDependency(node=agent_bank_documentation)],
    ).inputs(input=combine_inputs)

    workflow = Workflow(flow=Flow(nodes=[agent_bank_documentation, agent_bank_support]))
    result = workflow.run(input_data={"input": input})
    return result.output[agent_bank_support.id]["output"]["content"]


if __name__ == "__main__":
    print(run_workflow("fast block my card"))
