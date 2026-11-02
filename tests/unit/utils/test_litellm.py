import litellm
import pytest

from dynamiq.utils import litellm as lazy_litellm


def test_setup_step_runs_at_once_when_litellm_is_imported():
    calls = []

    lazy_litellm.on_litellm_import(lambda: calls.append("ran"))

    assert calls == ["ran"]


def test_failing_setup_step_is_logged_and_does_not_raise(mocker):
    logger = mocker.patch("dynamiq.utils.litellm.logger")

    def broken_step():
        raise RuntimeError("boom")

    lazy_litellm.on_litellm_import(broken_step)

    logger.warning.assert_called_once()
    assert "broken_step: boom" in logger.warning.call_args.args[0]
    assert lazy_litellm.get_litellm() is litellm


@pytest.mark.parametrize(
    ("wrapper", "target"),
    [
        ("token_counter", "litellm.token_counter"),
        ("get_max_tokens", "litellm.get_max_tokens"),
        ("get_model_info", "litellm.get_model_info"),
        ("get_supported_openai_params", "litellm.get_supported_openai_params"),
        ("supports_vision", "litellm.supports_vision"),
        ("supports_response_schema", "litellm.supports_response_schema"),
        ("supports_pdf_input", "litellm.utils.supports_pdf_input"),
        ("supports_prompt_caching", "litellm.utils.supports_prompt_caching"),
    ],
)
def test_wrappers_forward_to_litellm(mocker, wrapper, target):
    forwarded = mocker.patch(target, return_value="result")

    assert getattr(lazy_litellm, wrapper)("test-model", extra=1) == "result"
    forwarded.assert_called_once_with("test-model", extra=1)
