import io

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from examples.llm_setup import setup_llm

IMAGE_FILE = ""


def run_files_with_images_workflow():
    """Example workflow that automatically detects images in the files field"""
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=1)
    agent = ReActAgent(
        name="FilesImageAgent",
        id="files_image_agent",
        inference_mode=InferenceMode.XML,
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
    run_files_with_images_workflow()
