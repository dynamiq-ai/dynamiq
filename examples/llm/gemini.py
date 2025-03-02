"""
Integration of Gemini LLM into Dynamiq Framework

This script demonstrates how to integrate the Gemini Large Language Model (LLM) 
within the Dynamiq framework using a predefined prompt template. The model is 
configured with specific parameters to generate a short response based on the 
provided input text.

Dependencies:
- dynamiq

Ensure you have the necessary API key and dependencies installed before running the script.
"""

from dynamiq.nodes.llms.gemini import Gemini
from dynamiq.connections import Gemini as GeminiConnection
from dynamiq.prompts import Prompt, Message

# Define the prompt template for generating a short response
PROMPT_TEMPLATE = """
Write a short response on this topic: {{ text }}
"""

# Create a Prompt object with the defined template
prompt = Prompt(messages=[Message(content=PROMPT_TEMPLATE, role="user")])

# Configure the Gemini LLM Node
llm = Gemini(
    id="gemini",  # Unique identifier for the node
    connection=GeminiConnection(api_key=os.getenv('GEMINI_API_KEY')),  # Securely provide your API key
    model="gemini-1.5-flash",  # LLM model version
    temperature=0.3,  # Controls randomness (0.0 = deterministic, 1.0 = very random)
    max_tokens=1000,  # Maximum tokens in the generated response
    prompt=prompt  # Associated prompt
)

# Input data for the model
input_data = {
    "text": "Future of AI Agents"  # The topic for the model to generate a response on
}

# Execute the LLM node with the provided input
result = llm.run(input_data=input_data)

# Print the generated output
print(result.output)