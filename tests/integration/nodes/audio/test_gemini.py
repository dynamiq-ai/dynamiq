from io import BytesIO
from unittest.mock import ANY

import pytest

from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.nodes.audio.gemini import GeminiSTT
from dynamiq.runnables import RunnableResult, RunnableStatus


@pytest.mark.parametrize(
    ("audio", "mime_type"),
    [
        (b"bytes_data", "audio/wav"),
        (BytesIO(b"bytes_data"), "audio/wav"),
        (b"\xff\xfb\x90\xc4\x00\x00\n\xddu", "audio/mpeg"),
    ],
)
def test_workflow_with_gemini_transcriber(
    mock_gemini_transcript_response, mock_audio_transcribing_response_text, audio, mime_type
):
    model = "gemini/gemini-1.5-flash"
    connection = connections.Gemini(
        api_key="api-key",
    )
    wf_gemini = Workflow(
        flow=Flow(
            nodes=[GeminiSTT(name="gemini", connection=connection, model=model)],
        ),
    )
    input_data = {"audio": audio}
    response = wf_gemini.run(input_data=input_data)
    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"content": mock_audio_transcribing_response_text},
    ).to_dict(skip_format_types={BytesIO, bytes})

    expected_output = {wf_gemini.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
    assert response.output == expected_output
    mock_gemini_transcript_response.assert_called_once_with(
        model=model,
        messages=[
            {
                "content": [
                    {"text": "Generate a transcript of the speech.", "type": "text"},
                    {
                        "file": {
                            "file_data": ANY,
                        },
                        "type": "file",
                    },
                ],
                "role": "user",
            }
        ],
        api_key=connection.api_key,
    )
