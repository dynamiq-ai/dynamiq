import io
import uuid

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunStatus, RunType
from dynamiq.flows import Flow
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.images import ImageEdit, ImageResponseFormat, ImageSize
from dynamiq.runnables import RunnableConfig, RunnableStatus
from tests.integration.nodes.images.conftest import CREATION_TIMESTAMP


@pytest.fixture
def mock_mask_file(mock_image_bytes):
    """Create a test mask file."""
    mask_file = io.BytesIO(mock_image_bytes)
    mask_file.name = "test_mask.png"
    mask_file.seek(0)
    return mask_file


@pytest.mark.parametrize(
    ("response_format", "expected_content_type"),
    [
        (None, "url"),
        (ImageResponseFormat.B64_JSON, "b64_json"),
    ],
)
def test_image_edit_response_formats(
    mock_image_file,
    mock_image_edit_executor,
    mock_httpx_get,
    mock_image_url,
    response_format,
    expected_content_type,
):
    """Test image editing with different response formats."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")

    node = ImageEdit(
        name="Image Editor",
        model="gpt-image-1",
        connection=openai_connection,
        size=ImageSize.SIZE_1024x1024,
        response_format=response_format,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[node]),
    )

    input_data = {
        "prompt": "Add a sunset background",
        "files": mock_image_file,
    }

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[]),
    )

    assert response.status == RunnableStatus.SUCCESS
    node_output = response.output[node.id]
    assert "content" in node_output["output"]
    assert "files" in node_output["output"]
    assert len(node_output["output"]["files"]) == 1
    assert isinstance(node_output["output"]["files"][0], io.BytesIO)
    assert node_output["output"]["created"] == CREATION_TIMESTAMP

    mock_image_edit_executor.assert_called_once()
    call_kwargs = mock_image_edit_executor.call_args[1]

    if expected_content_type == "url":
        assert node_output["output"]["content"] == [mock_image_url]
        assert "response_format" not in call_kwargs or call_kwargs.get("response_format") is None
        mock_httpx_get.assert_called_once_with(mock_image_url, timeout=60.0)
    else:
        assert "test_image_0.png created" in node_output["output"]["content"][0]
        assert call_kwargs["response_format"] == ImageResponseFormat.B64_JSON


def test_image_edit_with_mask(
    mock_image_file,
    mock_mask_file,
    mock_image_edit_executor,
    mock_httpx_get,
):
    """Test image editing with a mask."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")

    node = ImageEdit(
        name="Image Editor",
        model="gpt-image-1",
        connection=openai_connection,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[node]),
    )

    input_data = {
        "prompt": "Change the masked area to blue",
        "files": mock_image_file,
        "mask": mock_mask_file,
    }

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[]),
    )

    assert response.status == RunnableStatus.SUCCESS

    call_kwargs = mock_image_edit_executor.call_args[1]
    assert "mask" in call_kwargs
    assert isinstance(call_kwargs["mask"], io.BytesIO)


@pytest.mark.parametrize("n_images", [1, 3])
def test_image_edit_multiple_outputs(
    mock_image_file,
    mock_image_edit_executor,
    mock_httpx_get,
    n_images,
):
    """Test editing to generate multiple variations."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")

    node = ImageEdit(
        name="Multi Image Editor",
        model="gpt-image-1",
        connection=openai_connection,
        n=n_images,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[node]),
    )

    input_data = {
        "prompt": f"{n_images} different edits",
        "files": mock_image_file,
    }

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[]),
    )

    assert response.status == RunnableStatus.SUCCESS
    node_output = response.output[node.id]
    assert len(node_output["output"]["content"]) == n_images
    assert len(node_output["output"]["files"]) == n_images
    assert all(isinstance(f, io.BytesIO) for f in node_output["output"]["files"])

    call_kwargs = mock_image_edit_executor.call_args[1]
    assert call_kwargs["n"] == n_images


@pytest.mark.parametrize(
    ("image_input", "expected_type"),
    [
        ("single_file", io.BytesIO),
        ("list_of_files", list),
        ("raw_bytes", bytes),
        ("empty_list", list),
        ("none", type(None)),
    ],
)
def test_image_edit_input_types(
    mock_image_bytes,
    mock_image_edit_executor,
    mock_httpx_get,
    image_input,
    expected_type,
):
    """Test editing with different input types."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")

    node = ImageEdit(
        name="Image Editor",
        model="gpt-image-1",
        connection=openai_connection,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[node]),
    )

    if image_input == "single_file":
        image_data = io.BytesIO(mock_image_bytes)
        image_data.name = "test.png"
        should_succeed = True
    elif image_input == "list_of_files":
        image1 = io.BytesIO(mock_image_bytes)
        image1.name = "image1.png"
        image2 = io.BytesIO(mock_image_bytes)
        image2.name = "image2.png"
        image_data = [image1, image2]
        should_succeed = True
    elif image_input == "raw_bytes":
        image_data = mock_image_bytes
        should_succeed = True
    elif image_input == "empty_list":
        image_data = []
        should_succeed = False
    else:
        image_data = None
        should_succeed = False

    input_data = {
        "prompt": "Edit image",
        "files": image_data,
    }

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[]),
    )

    if should_succeed:
        assert response.status == RunnableStatus.SUCCESS
        node_output = response.output[node.id]
        assert "files" in node_output["output"]
        call_kwargs = mock_image_edit_executor.call_args[1]
        assert "image" in call_kwargs
    else:
        assert response.status == RunnableStatus.FAILURE


def test_image_edit_with_tracing(
    mock_image_file,
    mock_image_edit_executor,
    mock_httpx_get,
):
    """Test image editing with tracing callback handler."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")

    node = ImageEdit(
        name="Image Editor",
        model="gpt-image-1",
        connection=openai_connection,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[node]),
    )

    tracing = TracingCallbackHandler()
    input_data = {
        "prompt": "Add dramatic lighting",
        "files": mock_image_file,
    }

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
    assert node_run.metadata["node"]["name"] == "Image Editor"
    assert node_run.status == RunStatus.SUCCEEDED


def test_image_edit_optimized_for_agents_with_tracing(
    mock_image_file,
    mock_image_edit_executor,
    mock_httpx_get,
):
    """Test image edit tool optimized for agents with tracing."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")
    tracing = TracingCallbackHandler()

    image_tool = ImageEdit(
        name="Edit Image",
        model="gpt-image-1",
        connection=openai_connection,
        is_optimized_for_agents=True,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[image_tool]),
    )

    input_data = {
        "prompt": "Add sunset background",
        "files": mock_image_file,
    }

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )

    assert response.status == RunnableStatus.SUCCESS
    assert len(tracing.runs) > 0

    node_output = response.output[image_tool.id]
    content = node_output["output"]["content"]
    assert isinstance(content, str)
    assert "## Edited Images" in content
    assert "Created:" in content
    assert "Files Generated" in content


def test_image_edit_preserves_original_filename(
    mock_image_file,
    mock_image_edit_executor,
    mock_httpx_get,
):
    """Test that edited images preserve original filename in output."""
    openai_connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="test-api-key")

    node = ImageEdit(
        name="Image Editor",
        model="gpt-image-1",
        connection=openai_connection,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[node]),
    )

    mock_image_file.name = "original_photo.jpg"

    input_data = {
        "prompt": "Brighten the image",
        "files": mock_image_file,
    }

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[]),
    )

    assert response.status == RunnableStatus.SUCCESS
    node_output = response.output[node.id]
    assert len(node_output["output"]["files"]) == 1
    output_file = node_output["output"]["files"][0]
    assert hasattr(output_file, "name")
    assert "original_photo" in output_file.name
