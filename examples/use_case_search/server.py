import os
import re

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import ScaleSerp
from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm


def extract_answer(text):
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return None


def extract_source(text):
    answer_match = re.search(r"<sources>(.*?)</sources>", text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return None


os.environ["SERP_API_KEY"] = os.getenv("SERP_API_KEY_DYN")
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
{search_results}
</search_results>
And here is the user's query:
<user_query>
{user_query}
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


llm_mini = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", max_tokens=500, temperature=0.5)
llm = setup_llm(model_provider="gpt", model_name="gpt-4o", max_tokens=3000, temperature=0.1)

agent_query_rephraser = SimpleAgent(
    id="agent_query_rephraser",
    name="agent_query_rephraser",
    role=AGENT_QUERY_ROLE,
    llm=llm_mini,
)

search_tool = ScaleSerpTool(
    name="search_tool",
    id="search_tool",
    connection=ScaleSerp(params={"location": "USA, United Kingdom, Europe"}),
    limit=5,
    is_optimized_for_agents=True,
).depends_on(agent_query_rephraser)
search_tool.input_transformer = InputTransformer(selector={"input": "$[agent_query_rephraser].output.content"})

agent_answer_synthesizer = SimpleAgent(
    id="agent_answer_synthesizer",
    name="agent_answer_synthesizer",
    role=AGENT_ANSWER_ROLE,
    llm=llm,
).depends_on([search_tool, agent_query_rephraser])
agent_answer_synthesizer._prompt_variables.update({"search_results": "{search_results}", "user_query": "{user_query}"})
agent_answer_synthesizer.input_transformer = InputTransformer(
    selector={
        "input": "",
        "search_results": "$[search_tool].output.content",
        "user_query": "$[agent_query_rephraser].output.content",
    }
)


tracing = TracingCallbackHandler()
wf = Workflow(flow=Flow(nodes=[agent_query_rephraser, search_tool, agent_answer_synthesizer]), callbacks=[tracing])


def process_query(query: str):
    """
    Generator that processes the query and streams the result chunk by chunk.
    This function yields each chunk to simulate real-time streaming of data.
    """
    # Run the workflow with the provided query
    result = wf.run(input_data={"input": query}, config=RunnableConfig(callbacks=[tracing]))

    result = result.output[agent_answer_synthesizer.id]["output"].get("content")
    logger.info(f"result: {result}")
    result_answer = extract_answer(result)
    result_sources = extract_source(result)

    result_answer_chunks = result_answer.split(" ")
    result_sources_chunks = [chunk + "\n\n" for chunk in result_sources.split("\n")]

    yield from ["Sources:\n\n"]
    yield from result_sources_chunks
    yield from ["\n\nAnswer:\n\n"]
    yield from result_answer_chunks
