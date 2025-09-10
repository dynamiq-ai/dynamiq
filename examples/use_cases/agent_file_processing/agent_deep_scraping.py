from dynamiq.connections import Http as HttpConnection
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.http_api_call import HttpApiCall, ResponseType
from dynamiq.nodes.types import InferenceMode
from dynamiq.storages.file import InMemoryFileStore
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

PORT = 5100

AGENT_ROLE = "A helpful and general-purpose AI assistant"

PROMPT1 = f"""Create test.txt file and upload it to the server. Return the content of the file.
            Send file to http://localhost:{PORT}/upload"""

if __name__ == "__main__":
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
        description="Tool to uploads a file to the server. Provide parameter file with ",
    )

    llm = setup_llm(model_provider="claude", model_name="claude-3-7-sonnet-20250219", temperature=0)

    agent = ReActAgent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[file_upload_api],
        role=AGENT_ROLE,
        filestorage=file_storage,
        max_loops=30,
        inference_mode=InferenceMode.FUNCTION_CALLING,
    )

    result = agent.run(input_data={"input": PROMPT1})

    output_content = result.output.get("content")
    logger.info("RESULT")
    logger.info(output_content)
