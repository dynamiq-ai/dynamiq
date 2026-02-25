import asyncio
import time

import streamlit as st

from dynamiq import Workflow
from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.connections import Tavily as TavilyConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools import TavilyTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm

AGENT_ROLE = "A helpful Assistant with access to web tools."

LLM_CONFIGS = [
    {"provider": "claude", "model": "claude-3-5-sonnet-20241022", "name": "GPT-4o mini"},
]

INFERENCE_MODES = [
    InferenceMode.FUNCTION_CALLING,
    InferenceMode.STRUCTURED_OUTPUT,
    InferenceMode.XML,
]

STREAMING_MODES = [
    StreamingMode.FINAL,
    StreamingMode.ALL,
]


def init_stream_ui(reset: bool = False):
    """Initialize or reset dynamic step boxes for streaming UI."""
    if reset or "step_placeholders" not in st.session_state:
        st.session_state.step_placeholders = {}
        st.session_state.step_contents = {}
        st.session_state.step_order = []
        st.session_state.current_config = None
        st.session_state.current_loop_nums = {}


def clear_stream_ui():
    """Clear all streaming UI elements."""
    st.session_state.step_placeholders = {}
    st.session_state.step_contents = {}
    st.session_state.step_order = []
    st.session_state.current_loop_nums = {}


def streamlit_callback(step: str, content):
    """Callback function to handle streaming content display."""
    if not content:
        return

    if step not in ["reasoning", "answer"]:
        return

    step_key = (step or "").lower().strip() or "unknown"
    display_name = step_key.replace("_", " ").title()

    # Initialize loop tracking if not exists
    if "current_loop_nums" not in st.session_state:
        st.session_state.current_loop_nums = {}

    # Handle the reasoning step
    if step == "reasoning" and isinstance(content, dict):
        thought = content.get("thought", "")
        loop_num = content.get("loop_num", 0)

        current_loop = st.session_state.current_loop_nums.get(step_key, -1)
        if current_loop != loop_num and current_loop != -1:
            # Add separator when loop number changes
            if step_key in st.session_state.step_contents:
                st.session_state.step_contents[step_key] += "\n\n"

        st.session_state.current_loop_nums[step_key] = loop_num
        content_to_display = thought
    else:
        # Handle the answer step
        content_to_display = str(content) if not isinstance(content, str) else content

    # Create a new box for a step on first occurrence
    if step_key not in st.session_state.step_placeholders:
        box = st.container(border=True)
        with box:
            st.markdown(f"### {display_name}")
            placeholder = st.empty()
        st.session_state.step_placeholders[step_key] = placeholder
        st.session_state.step_contents[step_key] = ""
        st.session_state.step_order.append(step_key)

    # Append the incoming content and update the box
    st.session_state.step_contents[step_key] += content_to_display
    st.session_state.step_placeholders[step_key].markdown(st.session_state.step_contents[step_key])


def run_agent(
    request: str,
    send_handler: AsyncStreamingIteratorCallbackHandler,
    llm_config: dict[str, str],
    inference_mode: InferenceMode,
    streaming_mode: StreamingMode,
) -> str:
    """
    Creates and runs agent with specified configuration.

    Args:
        request (str): The input request for the agent.
        send_handler (AsyncStreamingIteratorCallbackHandler): Handler of output messages.
        llm_config (Dict[str, str]): LLM configuration.
        inference_mode (InferenceMode): The inference mode to use.
        streaming_mode (StreamingMode): The streaming mode to use.

    Returns:
        str: Agent final output.
    """
    connection_tavily = TavilyConnection()
    tool_search = TavilyTool(connection=connection_tavily)

    try:
        llm = setup_llm(
            model_provider=llm_config["provider"], model_name=llm_config["model"], temperature=0, max_tokens=4000
        )
    except Exception as e:
        return f"Error setting up LLM: {str(e)}"

    research_agent = Agent(
        name="TestAgent",
        id="TestAgent",
        llm=llm,
        tools=[tool_search],
        role=AGENT_ROLE,
        inference_mode=inference_mode,
        streaming=StreamingConfig(enabled=True, mode=streaming_mode),
    )

    flow = Workflow(
        flow=Flow(nodes=[research_agent]),
    )

    try:
        response = flow.run(input_data={"input": request}, config=RunnableConfig(callbacks=[send_handler]))
        return response.output[research_agent.id]["output"]["content"]
    except Exception as e:
        return f"Error running agent: {str(e)}"


async def _send_stream_events_by_ws(send_handler):
    """Handle streaming events and update UI."""
    async for message in send_handler:
        data = message.data
        if not isinstance(data, dict):
            continue
        choices = data.get("choices") or []
        if not choices:
            continue
        delta = choices[-1].get("delta", {})
        step = delta.get("step", "")
        content = delta.get("content", "")
        if step and content:
            streamlit_callback(step, content)


def display_config_info(
    llm_config: dict[str, str],
    inference_mode: InferenceMode,
    streaming_mode: StreamingMode,
    iteration: int,
    total_iterations: int,
):
    """Display current configuration information."""
    st.markdown("---")
    st.header(f"🧪 Test Configuration ({iteration}/{total_iterations})")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🤖 LLM Configuration**")
        st.write(f"**Provider:** {llm_config['provider']}")
        st.write(f"**Model:** {llm_config['model']}")

    with col2:
        st.markdown("**🧠 Inference Mode**")
        st.write(f"**Mode:** {inference_mode.value}")

    with col3:
        st.markdown("**📡 Streaming Mode**")
        st.write(f"**Mode:** {streaming_mode.value.upper()}")

    st.markdown("---")


async def run_single_test(
    request: str,
    llm_config: dict[str, str],
    inference_mode: InferenceMode,
    streaming_mode: StreamingMode,
    iteration: int,
    total_iterations: int,
) -> str:
    """Run a single test configuration."""
    # Clear previous UI state
    clear_stream_ui()

    # Display configuration
    display_config_info(llm_config, inference_mode, streaming_mode, iteration, total_iterations)

    # Initialize streaming UI
    init_stream_ui(reset=True)

    # Create status container
    status_container = st.container()
    with status_container:
        st.info(f"🚀 Starting test with {llm_config['name']} - {inference_mode.value} - {streaming_mode.value.upper()}")

    # Setup async handler and task
    send_handler = AsyncStreamingIteratorCallbackHandler()
    current_loop = asyncio.get_running_loop()

    # Start streaming task
    streaming_task = current_loop.create_task(_send_stream_events_by_ws(send_handler))
    await asyncio.sleep(0.01)

    start_time = time.time()
    try:
        response = await current_loop.run_in_executor(
            None, run_agent, request, send_handler, llm_config, inference_mode, streaming_mode
        )

        await streaming_task

        end_time = time.time()
        duration = end_time - start_time

        with status_container:
            st.success(f"✅ Test completed in {duration:.2f}s")

        with st.expander("📋 Final Result", expanded=False):
            st.text_area("Agent Output", response, height=150)

        return response

    except Exception as e:
        with status_container:
            st.error(f"❌ Test failed: {str(e)}")
        return f"Test failed: {str(e)}"


async def run_all_tests():
    """Run all test configurations."""
    st.title("🧪 ReAct Agent Intermediate Streaming Testing")

    # Calculate total iterations
    total_iterations = len(LLM_CONFIGS) * len(INFERENCE_MODES) * len(STREAMING_MODES)

    st.info(f"🔢 Total configurations to test: **{total_iterations}**")

    # Add controls
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        auto_progress = st.checkbox("🔄 Auto-progress", value=False)

    with col2:
        delay_seconds = st.slider("⏱️ Delay (seconds)", 1, 10, 3) if auto_progress else 3

    with col3:
        test_question = st.text_input(
            "❓ Test Question",
            value="What is the weather like in New York today? Use search tools.",
            help="The question that will be asked to each configuration",
        )

    if not test_question:
        st.warning("Please enter a test question.")
        return

    # Initialize session state
    if "current_iteration" not in st.session_state:
        st.session_state.current_iteration = 0
        st.session_state.test_results = []

    # Add progress tracking and control buttons
    progress_bar = st.progress(0)
    progress_text = st.empty()

    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])

    with button_col1:
        start_tests = st.button("🚀 Start/Resume Tests", type="primary")

    with button_col2:
        reset_tests = st.button("🔄 Reset Tests")

    with button_col3:
        skip_current = st.button("⏭️ Skip Current")

    if reset_tests:
        st.session_state.current_iteration = 0
        st.session_state.test_results = []
        st.rerun()

    if skip_current and st.session_state.current_iteration < total_iterations:
        st.session_state.current_iteration += 1
        st.rerun()

    if start_tests or auto_progress:
        iteration = st.session_state.current_iteration

        if iteration >= total_iterations:
            st.success("🎉 All tests completed!")

            # Display summary
            st.header("📊 Test Summary")
            success_count = len([r for r in st.session_state.test_results if "Error" not in r.get("result", "")])
            st.metric("Successful Tests", f"{success_count}/{total_iterations}")

            return

        # Calculate current configuration
        llm_idx = iteration // (len(INFERENCE_MODES) * len(STREAMING_MODES))
        remaining = iteration % (len(INFERENCE_MODES) * len(STREAMING_MODES))
        inference_idx = remaining // len(STREAMING_MODES)
        streaming_idx = remaining % len(STREAMING_MODES)

        current_llm = LLM_CONFIGS[llm_idx]
        current_inference = INFERENCE_MODES[inference_idx]
        current_streaming = STREAMING_MODES[streaming_idx]

        # Update progress
        progress = iteration / total_iterations
        progress_bar.progress(progress)
        progress_text.text(f"Running test {iteration + 1} of {total_iterations}")

        result = await run_single_test(
            test_question, current_llm, current_inference, current_streaming, iteration + 1, total_iterations
        )

        # Store result
        test_result = {
            "iteration": iteration + 1,
            "llm": current_llm,
            "inference_mode": current_inference.value,
            "streaming_mode": current_streaming.value,
            "result": result,
            "timestamp": time.time(),
        }
        st.session_state.test_results.append(test_result)

        st.session_state.current_iteration += 1
        if auto_progress and st.session_state.current_iteration < total_iterations:
            with st.empty():
                for i in range(delay_seconds, 0, -1):
                    st.info(f"⏳ Next test starting in {i} seconds...")
                    await asyncio.sleep(1)
            st.rerun()
        elif st.session_state.current_iteration < total_iterations:
            st.info("👆 Click 'Start/Resume Tests' to continue to the next configuration.")


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Agent Configuration Tester", page_icon="🧪", layout="wide", initial_sidebar_state="collapsed"
    )

    # Initialize session state and run all tests
    init_stream_ui()
    asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()
