from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Http as HttpConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.http_api_call import HttpApiCall, ResponseType
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file import FileStoreConfig, InMemoryFileStore
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

PORT = 5000

AGENT_ROLE = "A helpful and general-purpose AI assistant"

PROMPT1 = f"""Create test.txt file and upload it to the server. Return the response of server.
            Send file to http://localhost:{PORT}/upload"""

if __name__ == "__main__":
    tracing = TracingCallbackHandler()
    connection = HttpConnection(
        method="POST",
        url=f"https://localhost:{PORT}/upload",
    )
    file_storage = InMemoryFileStore()

    file_upload_api = HttpApiCall(
        id="file-upload-api",
        connection=connection,
        response_type=ResponseType.JSON,
        name="FileUploadApi",
        description="Tool to upload a file to the server. Provide the parameter 'file' with the file to be uploaded. ",
    )

    llm = setup_llm(model_provider="claude", model_name="claude-3-7-sonnet-20250219", temperature=0)

    agent = ReActAgent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[file_upload_api],
        role=AGENT_ROLE,
        file_store=FileStoreConfig(enabled=True, backend=file_storage, agent_file_write_enabled=True),
        max_loops=30,
        inference_mode=InferenceMode.FUNCTION_CALLING,
    )

    wf = Workflow(
        flow=Flow(
            init_components=True,
            nodes=[agent],
        )
    )

    result = wf.run(input_data={"input": PROMPT1}, config=RunnableConfig(callbacks=[tracing]))

    output_content = result.output[agent.id]["output"].get("content")
    logger.info("RESULT")
    logger.info(output_content)
