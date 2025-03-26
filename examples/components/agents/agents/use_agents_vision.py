import io

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Exa
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from examples.llm_setup import setup_llm

IMAGE_URL = "https://media.istockphoto.com/id/1183183783/pl/zdj%C4%99cie/kobieta-artystka-pracuje-nad-abstrakcyjnym-malarstwem-olejnym-poruszaj%C4%85cym-p%C4%99dzlem.jpg?s=1024x1024&w=is&k=20&c=MG-drev2xDnvOFjl0tl5Zx2kL5LBW1bLTZYESx3qJFc="  # noqa E501
IMAGE_FILE = "../../data/img.jpeg"


def run_simple_agent_image_workflow():
    """Example workflow with image URL using SimpleAgent"""
    llm = setup_llm(model_provider="gemini", model_name="gemini-2.0-flash", temperature=1)
    agent = Agent(
        name="SimpleImageAgent",
        id="simple_image_agent",
        llm=llm,
    )

    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    result = wf.run(
        input_data={"input": "What can you tell me about this artwork?", "images": [IMAGE_URL]},
        config=RunnableConfig(callbacks=[tracing]),
    )

    agent_output = result.output[agent.id]["output"]["content"]
    print("SimpleAgent image workflow response:", agent_output)

    return agent_output, tracing.runs


def run_image_bytes_workflow():
    """Example workflow with image bytes input"""
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=1)
    agent = Agent(
        name="ImageBytesAgent",
        id="image_bytes_agent",
        llm=llm,
        inference_mode=InferenceMode.XML,
    )

    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    with open(IMAGE_FILE, "rb") as f:
        image_data = f.read()

    result = wf.run(
        input_data={"input": "Describe the image", "images": [image_data]},
        config=RunnableConfig(callbacks=[tracing]),
    )

    agent_output = result.output[agent.id]["output"]["content"]
    print("Image bytes workflow response:", agent_output)

    return agent_output, tracing.runs


def run_multiple_images_workflow():
    """Example workflow with multiple images of different types"""
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=1)
    agent = Agent(
        name="MultiImageAgent",
        id="multi_image_agent",
        llm=llm,
        inference_mode=InferenceMode.XML,
    )

    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    with open(IMAGE_FILE, "rb") as f:
        image1_data = f.read()

    result = wf.run(
        input_data={"input": "Compare these two images.", "images": [image1_data, IMAGE_URL]},
        config=RunnableConfig(callbacks=[tracing]),
    )

    agent_output = result.output[agent.id]["output"]["content"]
    print("Multiple images workflow response:", agent_output)

    return agent_output, tracing.runs


def run_tools_with_image_workflow():
    """Example workflow with tools and image input"""
    connection_exa = Exa()
    tool_search = ExaTool(connection=connection_exa)

    llm = setup_llm(model_provider="gpt", model_name="gpt-4.5-preview-2025-02-27", temperature=1)
    agent = Agent(
        name="ToolsImageAgent",
        id="tools_image_agent",
        llm=llm,
        tools=[tool_search],
        inference_mode=InferenceMode.XML,
    )

    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    with open(IMAGE_FILE, "rb") as f:
        image_data = f.read()

    result = wf.run(
        input_data={"input": "Find info about camera, just overall manufacturer", "images": [image_data]},
        config=RunnableConfig(callbacks=[tracing]),
    )

    agent_output = result.output[agent.id]["output"]["content"]
    print("Tools with image workflow response:", agent_output)

    return agent_output, tracing.runs


def run_files_with_images_workflow():
    """Example workflow that automatically detects images in the files field"""
    llm = setup_llm(model_provider="gpt", model_name="gpt-4.5-preview-2025-02-27", temperature=1)
    agent = Agent(
        name="FilesImageAgent",
        id="files_image_agent",
        llm=llm,
    )

    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    with open(IMAGE_FILE, "rb") as f:
        image_data = f.read()

    image_file = io.BytesIO(image_data)
    image_file.name = "uploaded_image.jpg"

    result = wf.run(
        input_data={"input": "Analyze what you see in the files I've uploaded", "files": [image_file]},
        config=RunnableConfig(callbacks=[tracing]),
    )

    agent_output = result.output[agent.id]["output"]["content"]
    print("Files with auto-detected images workflow response:", agent_output)

    return agent_output, tracing.runs


if __name__ == "__main__":
    run_simple_agent_image_workflow()
    # run_image_bytes_workflow()
    # run_multiple_images_workflow()
    # run_tools_with_image_workflow()
    # run_files_with_images_workflow()
