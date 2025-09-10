import re

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import ScaleSerp
from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm


def extract_tag_content(text, tag):
    """
    Extract content wrapped within specific XML-like tags from the text.

    Args:
        text (str): The input text containing the tag.
        tag (str): The tag name to extract content from.

    Returns:
        str: The content inside the tag if found, otherwise None.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


AGENT_QUERY_ROLE = """
You are an AI assistant tasked with processing and refactoring search queries.
Your goal is to rewrite queries to be more concise, clear, and useful for search engines. Follow these guidelines:
1. Remove unnecessary words like "what is," "who is," "where is," etc.
2. Convert questions into declarative statements
3. Focus on the core subject of the query
4. Maintain essential keywords
5. Ensure the refactored query is grammatically correct
Here are some examples of original queries and their refactored versions:
Original: "Who was the first person to walk on the moon?"
Refactored: "First person to walk on the moon"
Original: "What are the ingredients in a chocolate chip cookie?"
Refactored: "Chocolate chip cookie ingredients"
Original: "How tall is the Eiffel Tower?"
Refactored: "Eiffel Tower height"
Rewrite the query according to the guidelines provided. Output your refactored version, without any additional wording.
"""  # noqa E501

AGENT_ANSWER_ROLE = """
You are an AI assistant tasked with synthesizing answers from search results.
Your goal is to provide a concise and informative answer based on the following search results and user query.
Here are the search results:
<search_results>
{{search_results}}
</search_results>
And here is the user's query:
<user_query>
{{user_query}}
</user_query>
To complete this task, follow these steps:
1. Carefully read through the search results and identify the most relevant information that addresses the user's query.
2. Synthesize the information from multiple sources to create a comprehensive and accurate answer.
3. As you craft your answer, cite your sources using numbered markdown links formatted as [1], [2], etc. Each citation should correspond to a source in your source list.
4. Write your answer in a clear, concise, and journalistic tone. Avoid unnecessary jargon and explain any complex concepts in simple terms.
5. Ensure that your answer is well-structured and easy to read. Use paragraphs to separate different ideas or aspects of the answer.
6. After your main answer, provide a numbered list of sources. Format each source as follows:
   [number]. Source name (source URL)
Provide your synthesized answer within <answer> tags, followed by the source list within <sources> tags using link formatting like in markdown.
Your response should look like this:
<answer>
Your synthesized answer here, with citations like this [1] and this [2].
</answer>
<sources>
1. [Source Name 1](http://www.example1.com)
2. [Source Name 2](http://www.example2.com)
</sources>
Remember to focus on accuracy, clarity, and proper citation in your response.
If there are errors with the query or you are unable to craft a response, provide polite feedback within the <answer> tags.
Explain that you are not able to find the answer and provide some suggestions for the user to improve the query.
"""  # noqa E501

# Setup models
llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", max_tokens=500, temperature=0.5)

# Define agents and search tool
agent_query_rephraser = Agent(
    id="agent_query_rephraser",
    name="agent_query_rephraser",
    role=AGENT_QUERY_ROLE,
    llm=llm,
)

search_tool = ScaleSerpTool(
    name="search_tool",
    id="search_tool",
    connection=ScaleSerp(params={"location": "USA, United Kingdom, Europe"}),
    limit=5,
    is_optimized_for_agents=True,
    input_transformer=InputTransformer(selector={"query": f"$[{agent_query_rephraser.id}].output.content"}),
).depends_on(agent_query_rephraser)


agent_answer_synthesizer = OpenAI(
    id="agent_answer_synthesizer",
    name="agent_answer_synthesizer",
    connection=OpenAIConnection(),
    model="gpt-4o-mini",
    prompt=Prompt(
        messages=[
            Message(
                role="user",
                content=AGENT_ANSWER_ROLE,
            ),
        ],
    ),
    input_transformer=InputTransformer(
        selector={
            "input": "",
            "search_results": f"$[{search_tool.id}].output.content",
            "user_query": f"$[{agent_query_rephraser.id}].output.content",
        }
    ),
).depends_on([search_tool, agent_query_rephraser])

tracing = TracingCallbackHandler()
wf = Workflow(flow=Flow(nodes=[agent_query_rephraser, search_tool, agent_answer_synthesizer]), callbacks=[tracing])


def process_query(query: str):
    """
    Process the user's query through a workflow that rephrases, searches, and synthesizes an answer.
    Yields results chunk by chunk to simulate real-time streaming of data.

    Args:
        query (str): The original user query.

    Yields:
        str: Chunks of the final result (sources and answer).
    """
    try:
        # Run the workflow with the provided query
        result = wf.run(input_data={"input": query}, config=RunnableConfig(callbacks=[tracing]))

        output_content = result.output[agent_answer_synthesizer.id]["output"].get("content")
        logger.info(f"Workflow result: {output_content}")

        answer = extract_tag_content(output_content, "answer")
        sources = extract_tag_content(output_content, "sources")

        if answer and sources:
            # Stream the sources first
            yield "Sources:\n\n"
            for source_chunk in sources.split("\n"):
                yield source_chunk + "\n\n"

            # Stream the answer next
            yield "\n\nAnswer:\n\n"
            for answer_chunk in answer.split(" "):
                yield answer_chunk + " "
        else:
            yield "Error: Unable to extract answer or sources from the workflow output."

    except Exception as e:
        logger.error(f"An error occurred while processing the query: {e}")
        yield f"Error: {str(e)}"
