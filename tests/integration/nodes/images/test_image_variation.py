import io
import uuid

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunStatus, RunType
from dynamiq.flows import Flow
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.images import ImageResponseFormat, ImageSize, ImageVariation
from dynamiq.runnables import RunnableConfig, RunnableStatus


def test_image_variation_with_url_response(
    mock_image_file,
    mock_image_variation_executor,
    mock_httpx_get,
    mock_image_url,
):
    """Test basic image variation with URL response format."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")

    image_node = ImageVariation(
        name="Image Variator",
        model="gpt-image-1",
        connection=openai_connection,
        size=ImageSize.SIZE_1024x1024,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[image_node]),
    )

    input_data = {"image": mock_image_file}

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[]),
    )

    assert response.status == RunnableStatus.SUCCESS
    node_output = response.output[image_node.id]
    assert node_output["output"]["content"] == [mock_image_url]
    assert len(node_output["output"]["files"]) == 1
    assert isinstance(node_output["output"]["files"][0], io.BytesIO)
    assert node_output["output"]["created"] == 1234567890

    mock_image_variation_executor.assert_called_once()
    call_kwargs = mock_image_variation_executor.call_args[1]
    assert call_kwargs["model"] == "gpt-image-1"
    assert call_kwargs["size"] == "1024x1024"
    assert "image" in call_kwargs
    mock_httpx_get.assert_called_once_with(mock_image_url, timeout=60.0)


def test_image_variation_with_b64_response(
    mock_image_file,
    mock_image_variation_executor,
):
    """Test image variation with base64 JSON response format."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")

    image_node = ImageVariation(
        name="Image Variator B64",
        model="gpt-image-1",
        connection=openai_connection,
        size=ImageSize.SIZE_1024x1024,
        response_format=ImageResponseFormat.B64_JSON,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[image_node]),
    )

    input_data = {"image": mock_image_file}

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[]),
    )

    assert response.status == RunnableStatus.SUCCESS
    node_output = response.output[image_node.id]
    assert "test_image_0.png created" in node_output["output"]["content"][0]
    assert len(node_output["output"]["files"]) == 1
    assert isinstance(node_output["output"]["files"][0], io.BytesIO)

    mock_image_variation_executor.assert_called_once()
    call_kwargs = mock_image_variation_executor.call_args[1]
    assert call_kwargs["response_format"] == "b64_json"


def test_image_variation_with_list_of_images_uses_first(
    mock_image_bytes,
    mock_image_variation_executor,
    mock_httpx_get,
):
    """Test variation with a list of images (should use first one)."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")

    image1 = io.BytesIO(mock_image_bytes)
    image1.name = "image1.png"
    image2 = io.BytesIO(mock_image_bytes)
    image2.name = "image2.png"

    image_node = ImageVariation(
        name="Image Variator",
        model="gpt-image-1",
        connection=openai_connection,
        size=ImageSize.SIZE_1024x1024,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[image_node]),
    )

    input_data = {"image": [image1, image2]}

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[]),
    )

    assert response.status == RunnableStatus.SUCCESS
    node_output = response.output[image_node.id]
    assert "files" in node_output["output"]

    call_kwargs = mock_image_variation_executor.call_args[1]
    assert "image" in call_kwargs
    assert isinstance(call_kwargs["image"], io.BytesIO)


def test_image_variation_with_tracing(
    mock_image_file,
    mock_image_variation_executor,
    mock_httpx_get,
):
    """Test image variation with tracing callback handler."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")
    tracing = TracingCallbackHandler()

    image_node = ImageVariation(
        name="Image Variator",
        model="gpt-image-1",
        connection=openai_connection,
        size=ImageSize.SIZE_1024x1024,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[image_node]),
    )

    input_data = {"image": mock_image_file}

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
    assert node_run.metadata["node"]["name"] == "Image Variator"
    assert node_run.status == RunStatus.SUCCEEDED


def test_image_variation_optimized_for_agents_with_tracing(
    mock_image_file,
    mock_image_variation_executor,
    mock_httpx_get,
):
    """Test image variation tool optimized for agents with tracing."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")
    tracing = TracingCallbackHandler()

    image_tool = ImageVariation(
        name="Create Image Variations",
        model="gpt-image-1",
        connection=openai_connection,
        is_optimized_for_agents=True,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[image_tool]),
    )

    input_data = {"image": mock_image_file}

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )

    assert response.status == RunnableStatus.SUCCESS
    assert len(tracing.runs) > 0

    node_output = response.output[image_tool.id]
    content = node_output["output"]["content"]
    assert isinstance(content, str)
    assert "## Image Variations" in content
    assert "Created:" in content
    assert "Files Generated" in content


def test_image_variation_without_image_raises_error(
    mock_image_variation_executor,
):
    """Test that variation without an image returns a failure result."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")

    image_node = ImageVariation(
        name="Image Variator",
        model="gpt-image-1",
        connection=openai_connection,
        size=ImageSize.SIZE_1024x1024,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[image_node]),
    )

    input_data = {"image": None}

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[]),
    )

    assert response.status == RunnableStatus.FAILURE
