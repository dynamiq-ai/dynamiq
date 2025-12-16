import io
import uuid

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunStatus, RunType
from dynamiq.flows import Flow
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.images import ImageGeneration, ImageResponseFormat, ImageSize
from dynamiq.runnables import RunnableConfig, RunnableStatus


def test_image_generation_with_url_response(
    mock_image_generation_executor,
    mock_httpx_get,
    mock_image_url,
):
    """Test basic image generation with URL response format."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")

    image_node = ImageGeneration(
        name="Image Generator",
        model="gpt-image-1",
        connection=openai_connection,
        size=ImageSize.SIZE_1024x1024,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[image_node]),
    )

    input_data = {"prompt": "A beautiful sunset over mountains"}

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[]),
    )

    assert response.status == RunnableStatus.SUCCESS
    node_output = response.output[image_node.id]
    assert node_output["status"] == RunnableStatus.SUCCESS.value
    assert node_output["output"]["content"] == [mock_image_url]
    assert len(node_output["output"]["files"]) == 1
    assert isinstance(node_output["output"]["files"][0], io.BytesIO)

    mock_image_generation_executor.assert_called_once()
    call_kwargs = mock_image_generation_executor.call_args[1]
    assert call_kwargs["model"] == "gpt-image-1"
    assert call_kwargs["prompt"] == "A beautiful sunset over mountains"
    assert call_kwargs["size"] == "1024x1024"

    mock_httpx_get.assert_called_once_with(mock_image_url, timeout=60.0)


def test_image_generation_with_b64_response(
    mock_image_generation_executor,
    mock_image_b64,
):
    """Test image generation with base64 JSON response format."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")

    image_node = ImageGeneration(
        name="Image Generator B64",
        model="gpt-image-1",
        connection=openai_connection,
        size=ImageSize.SIZE_1024x1024,
        response_format=ImageResponseFormat.B64_JSON,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[image_node]),
    )

    input_data = {"prompt": "A serene lake at dawn"}

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[]),
    )

    assert response.status == RunnableStatus.SUCCESS
    node_output = response.output[image_node.id]
    assert node_output["output"]["content"] == ["image_0.png created"]
    assert len(node_output["output"]["files"]) == 1
    assert isinstance(node_output["output"]["files"][0], io.BytesIO)

    mock_image_generation_executor.assert_called_once()
    call_kwargs = mock_image_generation_executor.call_args[1]
    assert call_kwargs["response_format"] == "b64_json"


def test_image_generation_multiple_images(
    mock_image_generation_executor,
    mock_httpx_get,
    mock_image_url,
):
    """Test generating multiple images in one call."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")

    image_node = ImageGeneration(
        name="Multi Image Generator",
        model="gpt-image-1",
        connection=openai_connection,
        n=3,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[image_node]),
    )

    input_data = {"prompt": "Three variations of a cat"}

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[]),
    )

    assert response.status == RunnableStatus.SUCCESS
    node_output = response.output[image_node.id]
    assert len(node_output["output"]["content"]) == 3
    assert len(node_output["output"]["files"]) == 3
    assert all(isinstance(f, io.BytesIO) for f in node_output["output"]["files"])

    call_kwargs = mock_image_generation_executor.call_args[1]
    assert call_kwargs["n"] == 3


def test_image_generation_with_tracing(
    mock_image_generation_executor,
    mock_httpx_get,
):
    """Test image generation with tracing callback handler."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")
    tracing = TracingCallbackHandler()

    image_node = ImageGeneration(
        name="Image Generator",
        model="gpt-image-1",
        connection=openai_connection,
        size=ImageSize.SIZE_1024x1024,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[image_node]),
    )

    input_data = {"prompt": "A futuristic city"}

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )

    assert response.status == RunnableStatus.SUCCESS

    assert len(tracing.runs) > 0

    node_run = None
    for run in tracing.runs.values():
        if run.type == RunType.NODE and run.metadata.get("node", {}).get("group") == NodeGroup.IMAGES.value:
            node_run = run
            break

    assert node_run is not None
    assert node_run.metadata["node"]["name"] == "Image Generator"
    assert node_run.status == RunStatus.SUCCEEDED


def test_image_generation_optimized_for_agents_with_tracing(
    mock_image_generation_executor,
    mock_httpx_get,
):
    """Test image generation tool optimized for agents with tracing."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")
    tracing = TracingCallbackHandler()

    image_tool = ImageGeneration(
        name="Generate Image",
        model="gpt-image-1",
        connection=openai_connection,
        is_optimized_for_agents=True,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[image_tool]),
    )

    input_data = {"prompt": "A beautiful landscape"}

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )

    assert response.status == RunnableStatus.SUCCESS
    assert len(tracing.runs) > 0

    node_output = response.output[image_tool.id]
    content = node_output["output"]["content"]
    assert isinstance(content, str)
    assert "## Generated Images" in content
    assert "Created:" in content
    assert "Files Generated" in content
