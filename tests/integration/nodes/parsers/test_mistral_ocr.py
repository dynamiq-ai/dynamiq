import base64
from io import BytesIO
from unittest.mock import patch

import pytest
from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.runnables import RunnableResult, RunnableStatus

from dynamiq.connections import MistralOCR as MistralOCRConnection
from dynamiq.nodes.parsers import MistralOCR
from dynamiq.nodes.parsers.mistral_ocr import SUPPORTED_MIME_TYPES, MistralOCRInputSchema


def create_sample_image():
    """Create a simple test image as BytesIO object with JPEG header."""
    sample_data = BytesIO(b"\xff\xd8\xff\xe0sample image data")
    sample_data.name = "sample.jpg"
    return sample_data


def create_sample_pdf():
    """Create a simple test PDF as BytesIO object."""
    sample_data = BytesIO(b"%PDF-1.4\nsample pdf data")
    sample_data.name = "sample.pdf"
    return sample_data


def create_unsupported_file():
    """Create an unsupported file type as BytesIO object."""
    sample_data = BytesIO(b"Unsupported file content")
    sample_data.name = "sample.xyz"
    return sample_data


@pytest.mark.parametrize(
    ("file_input", "url", "mock_response", "setup_mocks", "expected_text"),
    [
        (
            "https://example.com/document.pdf",
            "https://api.mistral.ai/v1/ocr",
            {"pages": [{"markdown": "Sample OCR text from URL"}]},
            lambda rm, url, resp: rm.post(url=url, json=resp),
            "Sample OCR text from URL"
        ),
        (
            "https://example.com/sample.pdf",
            "https://api.mistral.ai/v1/ocr",
            {"pages": [{"markdown": "Different sample OCR text"}]},
            lambda rm, url, resp: rm.post(url=url, json=resp),
            "Different sample OCR text"
        ),
    ],
)
def test_workflow_with_mistral_ocr_urls(requests_mock, file_input, url, mock_response, setup_mocks, expected_text):
    wf_mistral_ocr = Workflow(
        flow=Flow(
            nodes=[
                MistralOCR(
                    connection=MistralOCRConnection(),
                )
            ]
        ),
    )

    input_data = {"file": file_input}
    call_mock = setup_mocks(requests_mock, url, mock_response)
    response = wf_mistral_ocr.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"text": expected_text},
    ).to_dict(skip_format_types={BytesIO})

    expected_output = {wf_mistral_ocr.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )

    assert call_mock.call_count == 1
    assert call_mock.last_request.url == url


@pytest.mark.parametrize(
    ("create_file_func", "expected_text"),
    [
        (create_sample_image, "Sample OCR text from image"),
        (create_sample_pdf, "Sample OCR text from PDF document"),
    ],
)
def test_workflow_with_mistral_ocr_files(requests_mock, create_file_func, expected_text):
    wf_mistral_ocr = Workflow(
        flow=Flow(
            nodes=[
                MistralOCR(
                    connection=MistralOCRConnection(),
                )
            ]
        ),
    )

    file_input = create_file_func()
    input_data = {"file": file_input}

    if create_file_func == create_sample_image:
        with patch("dynamiq.nodes.agents.utils.is_image_file", return_value=True), patch.object(
            MistralOCR, "_is_supported_file_type", return_value=True
        ):
            ocr_mock = requests_mock.post(
                url="https://api.mistral.ai/v1/ocr",
                json={"pages": [{"markdown": expected_text}]},
            )

            response = wf_mistral_ocr.run(input_data=input_data)

            assert ocr_mock.call_count == 1
            assert ocr_mock.last_request.url == "https://api.mistral.ai/v1/ocr"

    else:
        with patch("dynamiq.nodes.agents.utils.is_image_file", return_value=False), patch.object(
            MistralOCR, "_is_supported_file_type", return_value=True
        ):
            upload_mock = requests_mock.post(
                url="https://api.mistral.ai/v1/files",
                json={"id": "test-file-id-123", "purpose": "ocr"},
            )

            url_mock = requests_mock.get(
                url="https://api.mistral.ai/v1/files/test-file-id-123/url",
                json={"url": "https://api.mistral.ai/download/signed_pdf_url"},
            )

            head_mock = requests_mock.head(
                url="https://api.mistral.ai/download/signed_pdf_url",
                status_code=200,
            )

            ocr_mock = requests_mock.post(
                url="https://api.mistral.ai/v1/ocr",
                json={"pages": [{"markdown": expected_text}]},
            )

            response = wf_mistral_ocr.run(input_data=input_data)

            assert upload_mock.call_count == 1
            assert url_mock.call_count == 1
            assert ocr_mock.call_count == 1
            assert head_mock.call_count == 1
            assert upload_mock.last_request.url == "https://api.mistral.ai/v1/files"
            assert url_mock.last_request.url == "https://api.mistral.ai/v1/files/test-file-id-123/url?expiry=1"
            assert ocr_mock.last_request.url == "https://api.mistral.ai/v1/ocr"

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"text": expected_text},
    ).to_dict(skip_format_types={BytesIO})

    expected_output = {wf_mistral_ocr.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )


def test_mistral_ocr_with_base64_image():
    """Test MistralOCR with base64-encoded image data."""
    ocr = MistralOCR()

    image_data = b"\xff\xd8\xff\xe0sample image data"
    expected_text = "Extracted text from base64 image"

    base64_image = base64.b64encode(image_data).decode("utf-8")

    with patch.object(ocr.client, "request") as mock_request, patch.object(
        ocr, "_is_supported_file_type", return_value=True
    ):
        mock_request.return_value.status_code = 200
        mock_request.return_value.json.return_value = {"pages": [{"markdown": expected_text}]}

        image_io = BytesIO(image_data)

        with patch('dynamiq.nodes.agents.utils.is_image_file', return_value=True):
            result = ocr.execute(MistralOCRInputSchema(file=image_io))

        assert result["text"] == expected_text

        call_args = mock_request.call_args
        assert call_args[1]['method'] == 'POST'
        assert call_args[1]['url'].endswith('/v1/ocr')

        payload = call_args[1]['json']
        assert payload['document']['type'] == 'image_url'
        assert 'base64' in payload['document']['image_url']

        expected_base64_prefix = f"data:image/jpeg;base64,{base64_image}"
        assert payload["document"]["image_url"].startswith(expected_base64_prefix)


def test_mistral_ocr_unsupported_file_type():
    """Test that MistralOCR raises an error for unsupported file types."""
    ocr = MistralOCR()

    file_input = create_unsupported_file()

    with patch.object(ocr, "_is_supported_file_type", return_value=False):
        with pytest.raises(ValueError) as excinfo:
            ocr.execute(MistralOCRInputSchema(file=file_input))

        assert "Unsupported file type. OCR currently supports these mime types: " + ", ".join(
            SUPPORTED_MIME_TYPES
        ) in str(excinfo.value)


def test_supported_file_types():
    """Test the _is_supported_file_type method with various file types."""
    ocr = MistralOCR()

    with patch("dynamiq.nodes.agents.utils.is_image_file", return_value=True):
        assert ocr._is_supported_file_type(create_sample_image()) is True

    with patch("dynamiq.nodes.agents.utils.is_image_file", return_value=False):
        pdf_file = create_sample_pdf()
        assert ocr._is_supported_file_type(pdf_file) is True

    assert ocr._is_supported_file_type("https://example.com/document.pdf") is True

    with patch("dynamiq.nodes.agents.utils.is_image_file", return_value=False), patch(
        "mimetypes.guess_type", return_value=("application/octet-stream", None)
    ):
        assert ocr._is_supported_file_type(create_unsupported_file()) is False


def test_mistral_ocr_error_handling(requests_mock):
    """Test error handling in the MistralOCR node."""
    ocr = MistralOCR()
    url = "https://api.mistral.ai/v1/ocr"

    call_mock = requests_mock.post(
        url=url,
        status_code=401,
        json={"error": "Invalid API key"}
    )

    with patch.object(ocr, "_is_supported_file_type", return_value=True):
        with pytest.raises(ValueError) as excinfo:
            ocr.execute(MistralOCRInputSchema(file="https://example.com/document.pdf"))

        assert "Invalid API key" in str(excinfo.value)
        assert call_mock.call_count == 1
        assert call_mock.last_request.url == url


def test_extract_text_from_result():
    """Test the _extract_text_from_result method for different response formats."""
    ocr = MistralOCR()

    simple_result = {"pages": [{"markdown": "Simple extracted text"}]}
    assert ocr._extract_text_from_result(simple_result) == "Simple extracted text"

    pdf_result = {
        "pages": [
            {"markdown": "Page 1 content"},
            {"markdown": "Page 2 content"},
            {"markdown": "Page 3 content"},
        ]
    }
    assert ocr._extract_text_from_result(pdf_result) == "Page 1 content\n\nPage 2 content\n\nPage 3 content"
