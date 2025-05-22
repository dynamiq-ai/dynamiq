from dynamiq.connections import HttpApiKey
from dynamiq.nodes.llms.custom_llm import CustomLLM
from dynamiq.prompts import Prompt

OPENROUTER_API_KEY = ""
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "cognitivecomputations/dolphin3.0-mistral-24b:free"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.7

openrouter_connection = HttpApiKey(url=OPENROUTER_API_URL, api_key=OPENROUTER_API_KEY)

openrouter_llm = CustomLLM(
    name="OpenRouter",
    model=DEFAULT_MODEL,
    connection=openrouter_connection,
    max_tokens=DEFAULT_MAX_TOKENS,
    temperature=DEFAULT_TEMPERATURE,
    provider_prefix="openrouter",
)

prompt = Prompt(
    messages=[
        {"role": "system", "content": "You are helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
)

response = openrouter_llm.execute(input_data={}, prompt=prompt)
print(response["content"])
