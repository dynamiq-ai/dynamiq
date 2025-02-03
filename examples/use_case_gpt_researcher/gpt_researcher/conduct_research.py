from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import Pinecone, Tavily, ZenRows
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.nodes.operators import Map
from dynamiq.nodes.splitters.document import DocumentSplitter
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.tools.tavily import TavilyTool
from dynamiq.nodes.tools.zenrows import ZenRowsTool
from dynamiq.nodes.writers import PineconeDocumentWriter
from dynamiq.prompts import Message, Prompt
from dynamiq.storages.vector.pinecone.pinecone import PineconeIndexType
from examples.use_case_gpt_researcher.gpt_researcher.prompts import get_search_queries_prompt


def conduct_research_workflow():
    """Runs the research pipeline which saves web data related to user task in Pinecone storage."""

    # Perform initial search based on user query
    search_initial_query_node = TavilyTool(
        name="Tavily: Initial Query Search",
        id="search_initial_query_node",
        connection=Tavily(),
        input_transformer=InputTransformer(selector={"query": "$.query"}),
    )

    # Generate sub-queries from initial search results
    generate_sub_queries_node = OpenAI(
        name="OpenAI: Generate Sub-Queries",
        id="generate_sub_queries_node",
        model="gpt-4o-mini",
        connection=OpenAIConnection(),
        prompt=Prompt(messages=[Message(role="user", content=get_search_queries_prompt())]),
        temperature=1,
        input_transformer=InputTransformer(
            selector={
                "task": "$.query",
                "context": f"${[search_initial_query_node.id]}.output.content.result",
                "max_iterations": "$.max_iterations",
                "format": f"{', '.join([f'query {i + 1}' for i in range(3)])}",
            }
        ),
        depends=[NodeDependency(search_initial_query_node)],
    )

    # Parse generated sub-queries into a structured list
    parse_sub_queries_node = Python(
        name="Python: Parse Sub-Queries",
        id="parse_sub_queries_node",
        code="""
def run(input_data):
    try:
        result = input_data["queries"].strip('[').strip(']').split(',')
        result = [el.strip().strip('"').strip("'").strip() for el in result]
        result = [{"query": el} for el in result]
        return result
    except Exception as e:
        return input_data
        """,
        input_transformer=InputTransformer(selector={"queries": f"${[generate_sub_queries_node.id]}.output.content"}),
        depends=[NodeDependency(generate_sub_queries_node)],
    )

    # Execute searches for each sub-query
    search_sub_queries_node = Map(
        name="Tavily: Search Sub-Queries",
        id="search_sub_queries_node",
        node=TavilyTool(name="Tavily: Search Individual Sub-Query", connection=Tavily()),
        input_transformer=InputTransformer(selector={"input": f"${[parse_sub_queries_node.id]}.output.content"}),
        depends=[NodeDependency(parse_sub_queries_node)],
    )

    # Extract URLs from search results
    extract_links_node = Python(
        name="Python: Extract URLs",
        id="extract_links_node",
        code="""
def run(input_data):
    links = []
    for query_result in input_data['queries']:
        for el in query_result['content']['raw_response']['results']:
            links.append(el['url'])
    links = [{'url': el} for el in links]
    return links
        """,
        input_transformer=InputTransformer(selector={"queries": f"${[search_sub_queries_node.id]}.output.output"}),
        depends=[NodeDependency(search_sub_queries_node)],
    )

    # Retrieve web content from extracted links
    fetch_web_content_node = Map(
        name="ZenRows: Fetch Web Content",
        id="fetch_web_content_node",
        node=ZenRowsTool(connection=ZenRows()),
        input_transformer=InputTransformer(selector={"input": f"${[extract_links_node.id]}.output.content"}),
        depends=[NodeDependency(extract_links_node)],
    )

    # Convert fetched content into document format
    convert_to_documents_node = Python(
        name="Python: Convert to Documents",
        id="convert_to_documents_node",
        code="""
def run(input_data):
    from dynamiq.types import Document
    documents = []
    for el in input_data['content']['output']:
        try:
            doc = Document(content=el['content']['content'], metadata={"url": el['content']['url']})
            documents.append(doc)
        except Exception as e:
            pass
    return {'documents': documents}
        """,
        input_transformer=InputTransformer(selector={"content": f"${[fetch_web_content_node.id]}.output"}),
        depends=[NodeDependency(fetch_web_content_node)],
    )

    # Split documents into smaller segments
    split_documents_node = DocumentSplitter(
        name="Document Splitter",
        id="split_documents_node",
        split_by="character",
        split_length=1000,
        input_transformer=InputTransformer(
            selector={"documents": f"${[convert_to_documents_node.id]}.output.content.documents"}
        ),
        depends=[NodeDependency(convert_to_documents_node)],
    )

    # Embed documents for retrieval
    embed_documents_node = OpenAIDocumentEmbedder(
        name="OpenAI: Embed Documents",
        id="embed_documents_node",
        connection=OpenAIConnection(),
        input_transformer=InputTransformer(selector={"documents": f"${[split_documents_node.id]}.output.documents"}),
        depends=[NodeDependency(split_documents_node)],
    )

    # Store embedded documents in Pinecone
    store_documents_node = PineconeDocumentWriter(
        name="Pinecone: Store Documents",
        id="store_documents_node",
        connection=Pinecone(),
        index_name="gpt-researcher",
        index_type=PineconeIndexType.SERVERLESS,
        input_transformer=InputTransformer(selector={"documents": f"${[embed_documents_node.id]}.output.documents"}),
        depends=[NodeDependency(embed_documents_node)],
    )

    # Create workflow and add all nodes
    workflow = Workflow()
    for node in [
        search_initial_query_node,
        generate_sub_queries_node,
        parse_sub_queries_node,
        search_sub_queries_node,
        extract_links_node,
        fetch_web_content_node,
        convert_to_documents_node,
        split_documents_node,
        embed_documents_node,
        store_documents_node,
    ]:
        workflow.flow.add_nodes(node)

    return workflow
