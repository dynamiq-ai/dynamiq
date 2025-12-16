import io

import pytest

CREATION_TIMESTAMP = 1234567890


@pytest.fixture
def mock_image_url():
    return "https://example.com/generated_image.png"


@pytest.fixture
def mock_image_b64():
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="


@pytest.fixture
def mock_image_bytes(mock_image_b64):
    import base64

    return base64.b64decode(mock_image_b64)


@pytest.fixture
def mock_image_generation_executor(mocker, mock_image_url, mock_image_b64):
    """Mock for litellm.image_generation with URL and b64_json support."""

    def response(*args, **kwargs):
        from types import SimpleNamespace

        response_format = kwargs.get("response_format", "url")
        n = kwargs.get("n") or 1

        data = []
        for _ in range(n):
            if response_format == "b64_json":
                img_data = SimpleNamespace(b64_json=mock_image_b64, url=None)
            else:
                img_data = SimpleNamespace(url=mock_image_url, b64_json=None)
            data.append(img_data)

        return SimpleNamespace(
            data=data,
            created=CREATION_TIMESTAMP,
        )

    mock_gen = mocker.patch(
        "dynamiq.nodes.images.generation.ImageGeneration._image_generation",
        side_effect=response,
    )
    yield mock_gen


@pytest.fixture
def mock_image_edit_executor(mocker, mock_image_url, mock_image_b64):
    """Mock for litellm.image_edit with URL and b64_json support."""

    def response(*args, **kwargs):
        from types import SimpleNamespace

        response_format = kwargs.get("response_format", "url")
        n = kwargs.get("n") or 1

        data = []
        for _ in range(n):
            if response_format == "b64_json":
                img_data = SimpleNamespace(b64_json=mock_image_b64, url=None)
            else:
                img_data = SimpleNamespace(url=mock_image_url, b64_json=None)
            data.append(img_data)

        return SimpleNamespace(
            data=data,
            created=CREATION_TIMESTAMP,
        )

    mock_edit = mocker.patch(
        "dynamiq.nodes.images.edit.ImageEdit._image_edit",
        side_effect=response,
    )
    yield mock_edit


@pytest.fixture
def mock_image_variation_executor(mocker, mock_image_url, mock_image_b64):
    """Mock for litellm.image_variation with URL and b64_json support."""

    def response(*args, **kwargs):
        from types import SimpleNamespace

        response_format = kwargs.get("response_format", "url")
        n = kwargs.get("n") or 1
        data = []
        for _ in range(n):
            if response_format == "b64_json":
                img_data = SimpleNamespace(b64_json=mock_image_b64, url=None)
            else:
                img_data = SimpleNamespace(url=mock_image_url, b64_json=None)
            data.append(img_data)

        return SimpleNamespace(
            data=data,
            created=CREATION_TIMESTAMP,
        )

    mock_var = mocker.patch(
        "dynamiq.nodes.images.variation.ImageVariation._image_variation",
        side_effect=response,
    )
    yield mock_var


@pytest.fixture
def mock_httpx_get(mocker, mock_image_bytes):
    """Mock httpx.get for downloading images from URLs."""
    from types import SimpleNamespace

    def response(*args, **kwargs):
        return SimpleNamespace(
            content=mock_image_bytes,
            raise_for_status=lambda: None,
        )

    mock_get = mocker.patch("httpx.get", side_effect=response)
    yield mock_get


@pytest.fixture
def mock_image_file(mock_image_bytes):
    """Create a test image file."""
    image_file = io.BytesIO(mock_image_bytes)
    image_file.name = "test_image.png"
    image_file.seek(0)
    return image_file
