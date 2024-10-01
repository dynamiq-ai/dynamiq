import asyncio
import logging
from functools import partial

import chainlit as cl
from chainlit.input_widget import Select, Slider

from dynamiq import Workflow, flows, runnables
from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes import llms
from dynamiq.nodes.converters import UnstructuredFileConverter
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.nodes.retrievers import PineconeDocumentRetriever
from dynamiq.nodes.splitters.document import DocumentSplitter
from dynamiq.nodes.writers import PineconeDocumentWriter
from dynamiq.prompts import Message, Prompt
from dynamiq.storages.vector import PineconeVectorStore
from dynamiq.types.streaming import StreamingConfig
from dynamiq.utils import generate_uuid
from dynamiq.utils.logger import logger
from examples.chainlit.utils import default_prompt_template
from examples.rerankers.bge.node.bge import BGEDocumentRanker

logger.setLevel(logging.INFO)


def build_indexing_flow(vector_store: PineconeVectorStore):

    # initialize indexing nodes
    file_converter_node = UnstructuredFileConverter(strategy="hi_res")
    document_splitter_node = DocumentSplitter(
        split_by="title",
        split_length=2,
        depends=[
            NodeDependency(file_converter_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[file_converter_node.id]}.output.documents",
            },
        ),
    )
    document_embedder_node = OpenAIDocumentEmbedder(
        model="text-embedding-3-small",
        depends=[
            NodeDependency(document_splitter_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[document_splitter_node.id]}.output.documents",
            },
        ),
    )
    document_writer_node = PineconeDocumentWriter(
        vector_store=vector_store,
        depends=[
            NodeDependency(document_embedder_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[document_embedder_node.id]}.output.documents",
            },
        ),
    )

    # build the indexing flow
    indexing_flow = Flow(
        id=generate_uuid(),
        nodes=[
            file_converter_node,
            document_splitter_node,
            document_embedder_node,
            document_writer_node,
        ],
    )

    return indexing_flow


def build_inference_flow(vector_store: PineconeVectorStore, settings: dict):

    llm_model = settings.get("model", "gpt-3.5-turbo")
    temperature = settings.get("temperature", 0.0)
    number_of_sections = int(settings.get("number_of_sections", 5))

    text_embedder_node = OpenAITextEmbedder(model="text-embedding-3-small")
    document_retriever_node = PineconeDocumentRetriever(
        vector_store=vector_store,
        top_k=number_of_sections,
        depends=[
            NodeDependency(text_embedder_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "embedding": f"${[text_embedder_node.id]}.output.embedding",
            },
        ),
    )
    document_ranker_node = BGEDocumentRanker(
        model_name="BAAI/bge-reranker-v2-m3",
        threshold=0.0,
        top_k=number_of_sections,
        depends=[
            NodeDependency(document_retriever_node),
            NodeDependency(text_embedder_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[document_retriever_node.id]}.output.documents",
                "query": f"${[text_embedder_node.id]}.output.query",
            },
        ),
    )

    # intitalize the llm exectutor node
    message = Message(content=default_prompt_template(), role="user")
    prompt = Prompt(id=generate_uuid(), messages=[message])

    answer_generation_node = llms.OpenAI(
        id=generate_uuid(),
        name="OpenAI Answer Generation",
        model=llm_model,
        temperature=temperature,
        prompt=prompt,
        connection=OpenAIConnection(),
        streaming=StreamingConfig(enabled=True),
        depends=[
            NodeDependency(document_ranker_node),
        ],
        input_transformer=InputTransformer(
            selector={
                "documents": f"${[document_ranker_node.id]}.output.documents",
            },
        ),
    )

    # build the retrieval flow
    inference_workflow = Workflow(
        id=generate_uuid(),
        flow=flows.Flow(
            id=generate_uuid(),
            nodes=[
                text_embedder_node,
                document_retriever_node,
                document_ranker_node,
                answer_generation_node,
            ],
        ),
    )

    cl.user_session.set("ranker_node_id", document_ranker_node.id)

    return inference_workflow


@cl.on_chat_start
async def start_chat():

    # Step 0: Define settings
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="OpenAI - Model",
                values=[
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-16k",
                    "gpt-4",
                    "gpt-4-32k",
                    "gpt-4o",
                ],
                initial_index=0,
            ),
            Slider(
                id="temperature",
                label="OpenAI - Temperature",
                initial=0,
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id="number_of_sections",
                label="Search - Number of sections",
                initial=5,
                min=0,
                max=10,
                step=1,
            ),
        ]
    ).send()

    # Step 1: Ask the user to upload a file
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload an example file to begin!",
            accept=["application/pdf", "text/html"],
            max_size_mb=200,
            timeout=180,
        ).send()

    file = files[0]

    if file.type == "application/pdf":
        view_content_elements = [cl.Pdf(name=file.name, display="side", path=file.path)]
    else:
        view_content_elements = [
            cl.File(name=file.name, path=file.path, display="side")
        ]

    msg = cl.Message(
        content=f"Click on {file.name} to view the content. Started processing.",
        elements=view_content_elements,
    )
    await msg.send()

    # initialize the vector store
    vector_store = PineconeVectorStore(index_name="default", dimension=1536)
    if vector_store.count_documents() > 0:
        vector_store.delete_documents(delete_all=True)

    indexing_flow = build_indexing_flow(vector_store)

    async with cl.Step(name="File processing") as step:
        step.input = file.name
        output = indexing_flow.run(
            input_data={
                "file_paths": [file.path],
            }
        )
        step.output = output

    msg.content = (
        f"Click on {file.name} to view the content. File processed successfully! 🎉"
    )
    await msg.update()

    inference_flow = build_inference_flow(vector_store, settings)

    cl.user_session.set("inference_flow", inference_flow)
    cl.user_session.set("vector_store", vector_store)
    cl.user_session.set("settings", settings)


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("config", settings)
    vector_store = cl.user_session.get("vector_store")
    inference_flow = build_inference_flow(vector_store, settings)
    cl.user_session.set("inference_flow", inference_flow)


async def run_wf_async(
    wf: Workflow, wf_data: dict, streaming: AsyncStreamingIteratorCallbackHandler
) -> None:
    wf_run = partial(
        wf.run,
        input_data=wf_data,
        config=runnables.RunnableConfig(callbacks=[streaming]),
    )
    asyncio.get_running_loop().run_in_executor(None, wf_run)


@cl.on_message
async def main(message: cl.Message):
    inference_flow = cl.user_session.get("inference_flow")

    msg = cl.Message(content="")
    await msg.send()

    streaming = AsyncStreamingIteratorCallbackHandler()

    query = message.content
    logger.info(f"Running inference flow with message: {query}")

    # Run in async mode to avoid blocking the main thread
    await asyncio.create_task(run_wf_async(inference_flow, {"query": query}, streaming))
    await asyncio.sleep(0.01)

    final_output = {}
    async for event in streaming:
        if event.entity_id != inference_flow.id:
            if token_data := event.data["choices"][0]["delta"]["content"]:
                # All streaming events without final with full output
                await msg.stream_token(token_data)
        else:
            # Final event with full output
            final_output = event.data

    await msg.update()

    await asyncio.sleep(0.1)

    ranker_node_id = cl.user_session.get("ranker_node_id", "")
    retrieved_documents = (
        final_output.get(ranker_node_id, {}).get("output", {}).get("documents", [])
    )

    logger.info(f"Retrieved documents: {len(retrieved_documents)}")

    source_elements = []
    source_display_text = "Source context:\n"

    for i, doc in enumerate(retrieved_documents):
        score = round(float(doc.get("score", 0)), 2)

        source_elements.append(cl.Text(name=f"Section_{i}", content=doc.get("content", ""), display="side"))
        source_display_text += f"- Score: {score} Section_{i}\n"

    msg = cl.Message(content=source_display_text, elements=source_elements)
    await msg.send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
