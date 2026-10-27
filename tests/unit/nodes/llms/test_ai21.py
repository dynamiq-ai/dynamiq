import pytest

from dynamiq.connections import AI21 as AI21Connection
from dynamiq.nodes.llms import AI21
from dynamiq.utils.litellm import get_litellm


@pytest.mark.parametrize("model", ["jamba-large", "ai21/jamba-large", "ai21_chat/jamba-large"])
def test_ai21_models_go_to_ai21(model):
    llm = AI21(connection=AI21Connection(api_key="test"), model=model)

    request_model = llm.update_completion_params({"model": llm.model})["model"]
    _, provider, _, api_base = get_litellm().get_llm_provider(request_model)

    # Saved as earlier releases save it, so they can run the workflow too.
    assert llm.model == "ai21/jamba-large"
    assert provider == "ai21_chat"
    assert api_base.startswith("https://api.ai21.com/")
