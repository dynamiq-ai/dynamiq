import asyncio
import io
import os

import streamlit as st

from dynamiq import Workflow
from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.connections import E2B
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import Anthropic as AnthropicLLM
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from dynamiq.utils.logger import logger

AGENT_ROLE = """
You are a Senior Data Analyst with expertise in Python data analysis and visualization.
You have access to E2B sandbox tools to analyze CSV files and generate insights.
EFFICIENCY FIRST - MINIMIZE ITERATIONS:
    - COMBINE multiple analysis tasks into SINGLE tool calls to reduce iterations
    - Perform data loading, cleaning, analysis, and visualization in ONE comprehensive script
    - Generate ALL required visualizations and save them in ONE execution
    - Batch related operations together (e.g., load data + explore + visualize + analyze)
    - Use variables to store intermediate results and reuse them across different analyses
    - Avoid multiple separate tool calls for related tasks
Follow these rules:
    - ALWAYS FORMAT YOUR RESPONSE IN MARKDOWN.
    - Use pandas, matplotlib, seaborn, plotly, and other data analysis libraries.
    - Generate visualizations and statistical summaries.
    - Provide actionable insights from the data.
    - Save any generated plots and analysis results as files.
IMPORTANT FOR IMAGE HANDLING:
    - When you generate visualizations (plots, charts, graphs), ALWAYS save them as image files.
    - Use descriptive filenames like "sales_trend.png", "correlation_heatmap.png", "distribution_histogram.png".
    - Save images in PNG format for best quality: plt.savefig('filename.png', dpi=300, bbox_inches='tight').
    - In your final answer, reference saved images using the format [filename.png] - this will display them inline.
    - For example: "Here's the sales trend analysis: [sales_trend.png]"
    - The system will automatically display these images in your response.
VISUALIZATION BEST PRACTICES:
    - Create clear, professional-looking plots with proper titles and labels.
    - Use appropriate color schemes and styling.
    - Ensure plots are readable and informative.
    - Save multiple types of visualizations (line plots, bar charts, heatmaps, scatter plots, etc.).
    - Always include context and interpretation for each visualization.
EFFICIENCY EXAMPLES:
    - Instead of separate calls for "load data" then "create visualizations", do both in one script
    - Generate correlation matrix, distribution plots, and trend analysis in a single execution
    - Combine data quality assessment with visualization creation
    - Perform statistical tests and create corresponding visualizations together
"""


def init_stream_ui(reset: bool = False):
    """Initialize or reset dynamic step boxes for streaming UI."""
    if reset or "step_placeholders" not in st.session_state:
        st.session_state.step_placeholders = {}
        st.session_state.step_contents = {}
        st.session_state.step_order = []
        st.session_state.current_loop_nums = {}


def _show_loading_spinner():
    """Show a loading spinner while the agent is thinking."""
    import streamlit as st

    if "loading_placeholder" not in st.session_state:
        st.session_state.loading_placeholder = None
    if "is_loading" not in st.session_state:
        st.session_state.is_loading = False

    if st.session_state.loading_placeholder is None:
        st.session_state.loading_placeholder = st.empty()

    with st.session_state.loading_placeholder.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(
                """
            <div style="display: flex; align-items: center; justify-content: center; height: 40px;">
                <div style="width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid #3498db;
                 border-radius: 50%; animation: spin 1s linear infinite;"></div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown("🤖 **Agent is analyzing your data...**")
            st.markdown("*This may take a few moments*")

    st.markdown(
        """
    <style>
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def _hide_loading_spinner():
    """Hide the loading spinner."""
    import streamlit as st

    if "loading_placeholder" not in st.session_state:
        st.session_state.loading_placeholder = None
    if "is_loading" not in st.session_state:
        st.session_state.is_loading = False

    if st.session_state.loading_placeholder is not None:
        st.session_state.loading_placeholder.empty()
        st.session_state.loading_placeholder = None


def _convert_base64_to_bytesio(base64_string, file_name="file", content_type="unknown"):
    """Convert base64 string back to BytesIO object."""
    import base64
    import io

    try:
        content_bytes = base64.b64decode(base64_string)

        bytesio_obj = io.BytesIO(content_bytes)
        bytesio_obj.name = file_name
        bytesio_obj.content_type = content_type

        return bytesio_obj
    except Exception:
        return None


def _is_valid_base64(s):
    """Check if a string is valid base64."""
    import base64

    try:
        if isinstance(s, str):
            decoded = base64.b64decode(s, validate=True)
            if len(decoded) > 0 and len(decoded) < 100 * 1024 * 1024:  # Max 100MB
                return True
        return False
    except Exception:
        return False


def _process_tool_files_with_conversion(tool_files):
    """Process tool files, handling BytesIO objects, base64 strings, and SerializationIterator objects."""
    if not tool_files or not isinstance(tool_files, list):
        return []

    processed_files = []
    for i, file_obj in enumerate(tool_files):
        if hasattr(file_obj, "iterator") and hasattr(file_obj, "index"):
            bytesio_obj = file_obj.iterator
            if hasattr(bytesio_obj, "getvalue"):
                file_name = getattr(bytesio_obj, "name", f"file_{i}")
                file_content = bytesio_obj.getvalue()
                file_type = getattr(bytesio_obj, "content_type", "unknown")

                import base64

                base64_content = base64.b64encode(file_content).decode("utf-8")

                file_info = {
                    "name": file_name,
                    "content": base64_content,
                    "type": file_type,
                    "original_size": len(file_content),
                }

                processed_files.append(file_info)

        elif isinstance(file_obj, dict) and "content" in file_obj and _is_valid_base64(file_obj["content"]):
            file_name = file_obj.get("name", f"file_{i}")
            content_type = file_obj.get("content_type", "unknown")
            base64_content = file_obj["content"]
            description = file_obj.get("description", "")
            original_size = file_obj.get("original_size", 0)

            file_info = {
                "name": file_name,
                "content": base64_content,
                "type": content_type,
                "description": description,
                "original_size": original_size,
            }

            processed_files.append(file_info)

        elif isinstance(file_obj, str) and _is_valid_base64(file_obj):
            file_name = f"file_{i}"
            content_type = "unknown"

            bytesio_obj = _convert_base64_to_bytesio(file_obj, file_name, content_type)
            if bytesio_obj:
                file_content = bytesio_obj.getvalue()
                base64_content = file_obj

                file_info = {
                    "name": file_name,
                    "content": base64_content,
                    "type": content_type,
                    "original_size": len(file_content),
                }

                processed_files.append(file_info)

        elif hasattr(file_obj, "name") and hasattr(file_obj, "getvalue"):
            file_name = getattr(file_obj, "name", f"file_{i}")
            file_content = file_obj.getvalue()
            file_type = getattr(file_obj, "content_type", "unknown")

            import base64

            base64_content = base64.b64encode(file_content).decode("utf-8")

            file_info = {
                "name": file_name,
                "content": base64_content,
                "type": file_type,
                "original_size": len(file_content),
            }

            processed_files.append(file_info)
        elif isinstance(file_obj, str):
            continue

    return processed_files


def streamlit_callback(step: str, content):
    """Callback function to handle streaming content display with tool input and file display support."""
    if not content:
        return

    if step not in ["reasoning", "answer", "tool"]:
        return

    if "step_placeholders" not in st.session_state:
        st.session_state.step_placeholders = {}
    if "step_contents" not in st.session_state:
        st.session_state.step_contents = {}
    if "step_order" not in st.session_state:
        st.session_state.step_order = []
    if "current_loop_nums" not in st.session_state:
        st.session_state.current_loop_nums = {}
    if "generated_files" not in st.session_state:
        st.session_state.generated_files = []
    elif not isinstance(st.session_state.generated_files, list):
        st.session_state.generated_files = []
    if "loading_placeholder" not in st.session_state:
        st.session_state.loading_placeholder = None
    if "is_loading" not in st.session_state:
        st.session_state.is_loading = False

    step_key = "reasoning"
    display_name = "Reasoning"

    if step == "reasoning" and isinstance(content, dict):
        if st.session_state.is_loading:
            _hide_loading_spinner()
            st.session_state.is_loading = False

        thought = content.get("thought", "")
        loop_num = content.get("loop_num", 0)

        current_loop = st.session_state.current_loop_nums.get(step_key, -1)
        if current_loop != loop_num and current_loop != -1:
            if step_key in st.session_state.step_contents:
                st.session_state.step_contents[step_key] += "\n\n"

        st.session_state.current_loop_nums[step_key] = loop_num
        content_to_display = thought

    elif step == "tool" and isinstance(content, dict):
        if st.session_state.is_loading:
            _hide_loading_spinner()
            st.session_state.is_loading = False

        tool_name = content.get("name", "Unknown Tool")
        tool_input = content.get("input", "")
        tool_files = content.get("files", [])

        if "e2b" in tool_name.lower() or "interpreter" in tool_name.lower():
            content_to_display = ""

            if tool_input:
                if isinstance(tool_input, str):
                    try:
                        import json

                        tool_input_dict = json.loads(tool_input)
                        python_code = tool_input_dict.get("python", "")
                    except (json.JSONDecodeError, AttributeError):
                        python_code = tool_input if "import " in tool_input or "print(" in tool_input else ""
                elif isinstance(tool_input, dict):
                    python_code = tool_input.get("python", "")
                else:
                    python_code = ""

                if python_code:
                    content_to_display += "\n\n**🐍 Generated Python Code:**\n\n"
                    content_to_display += f"```python\n{python_code}\n```\n\n"

            processed_files = _process_tool_files_with_conversion(tool_files)

            if processed_files:
                for file_info in processed_files:
                    st.session_state.generated_files.append(file_info)
        else:
            return

    elif step == "answer":
        if st.session_state.is_loading:
            _hide_loading_spinner()
            st.session_state.is_loading = False

        step_key = "answer"
        display_name = "Answer"
        content_to_display = str(content) if not isinstance(content, str) else content

    else:
        content_to_display = str(content) if not isinstance(content, str) else content

    if step_key not in st.session_state.step_placeholders:
        if step_key == "reasoning":
            with st.expander("🤔 Reasoning", expanded=False):
                placeholder = st.empty()
        else:
            box = st.container(border=True)
            with box:
                st.markdown(f"### {display_name}")
                if step_key == "answer":
                    placeholder = st.empty()
                else:
                    placeholder = st.empty()
        st.session_state.step_placeholders[step_key] = placeholder
        st.session_state.step_contents[step_key] = ""
        st.session_state.step_order.append(step_key)

    st.session_state.step_contents[step_key] += content_to_display

    if step_key == "answer":
        processed_content = _process_file_references_in_text(st.session_state.step_contents[step_key])
        st.session_state.step_placeholders[step_key].markdown(processed_content)
    else:
        st.session_state.step_placeholders[step_key].markdown(st.session_state.step_contents[step_key])


def _display_all_files_at_end():
    """Display all generated files at the end of execution."""
    generated_files = st.session_state.get("generated_files", [])

    if not generated_files:
        return

    st.markdown("---")
    st.markdown("## 📁 Generated Files")
    st.markdown(f"**{len(generated_files)} files generated during analysis:**")

    for i, file_info in enumerate(generated_files):
        st.markdown(f"### {i+1}. {file_info.get('name', f'file_{i}')}")

        # Display file info
        file_type = file_info.get("type", "unknown")
        original_size = file_info.get("original_size", 0)
        description = file_info.get("description", "")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Type:** {file_type}")
            if original_size > 0:
                st.markdown(f"**Size:** {original_size:,} bytes")
            if description:
                st.markdown(f"**Description:** {description}")

        with col2:
            # Display the file based on its type
            try:
                if file_type.lower() in ["csv", "text/csv"]:
                    _display_csv_file_inline(
                        file_info.get("content", ""), file_info.get("name", f"file_{i}"), f"end_{i}"
                    )
                elif file_type.lower() in ["png", "jpg", "jpeg", "gif", "bmp", "image/png", "image/jpeg", "image/gif"]:
                    _display_image_file_inline(
                        file_info.get("content", ""), file_info.get("name", f"file_{i}"), f"end_{i}"
                    )
                elif file_type.lower() in ["json", "application/json"]:
                    _display_json_file_inline(
                        file_info.get("content", ""), file_info.get("name", f"file_{i}"), f"end_{i}"
                    )
                elif file_type.lower() in ["txt", "text/plain"]:
                    _display_text_file_inline(
                        file_info.get("content", ""), file_info.get("name", f"file_{i}"), f"end_{i}"
                    )
                else:
                    # Default: show as downloadable file
                    _display_single_file_inline(file_info, i)
            except Exception as e:
                st.error(f"Error displaying file {file_info.get('name', f'file_{i}')}: {e}")

        st.markdown("---")


def _process_file_references_in_text(text):
    """Process file references in text and replace with actual file content."""
    import re

    # Pattern to match file references like [filename.ext] or [filename]
    file_pattern = r"\[([^\]]+\.(?:csv|png|jpg|jpeg|gif|bmp|json|txt|pdf|html|xml|yaml|yml))\](?!\])"

    def replace_file_reference(match):
        file_name = match.group(1)

        # Find the file in generated files
        file_info = None
        generated_files = st.session_state.get("generated_files", [])

        for f in generated_files:
            stored_name = f.get("name", "")

            # Try exact match first, then endswith, then contains
            exact_match = stored_name == file_name
            endswith_match = stored_name.endswith(file_name)
            contains_match = file_name in stored_name
            path_match_forward = stored_name.split("/")[-1] == file_name
            path_match_backward = stored_name.split("\\")[-1] == file_name

            if exact_match or endswith_match or contains_match or path_match_forward or path_match_backward:
                file_info = f
                break

        if file_info:
            # Return the file content as actual content instead of placeholders
            file_content = file_info.get("content", "")
            file_type = file_info.get("type", "unknown")

            if file_type.lower() in ["png", "jpg", "jpeg", "gif", "bmp", "image/png", "image/jpeg", "image/gif"]:
                # For images, create a data URI
                if isinstance(file_content, str) and not file_content.startswith("data:"):
                    data_uri = f"data:image/png;base64,{file_content}"
                else:
                    data_uri = file_content
                return f"![{file_name}]({data_uri})"
            elif file_type.lower() in ["csv", "text/csv"]:
                # For CSV, show as a table preview
                try:
                    import base64
                    import io

                    import pandas as pd

                    # Convert base64 content to string if needed
                    if isinstance(file_content, str):
                        try:
                            content_bytes = base64.b64decode(file_content)
                            content = content_bytes.decode("utf-8")
                        except Exception:
                            content = file_content
                    elif isinstance(file_content, bytes):
                        content = file_content.decode("utf-8")
                    else:
                        content = str(file_content)

                    # Read CSV and create a preview
                    df = pd.read_csv(io.StringIO(content))
                    preview = df.head(5).to_string(index=False)
                    return f"**📊 {file_name}**\n```\n{preview}\n```"
                except Exception:
                    return f"**📊 {file_name}** (CSV file)"
            elif file_type.lower() in ["json", "application/json"]:
                # For JSON, show formatted content
                try:
                    import base64
                    import json

                    # Convert base64 content to string if needed
                    if isinstance(file_content, str):
                        try:
                            content_bytes = base64.b64decode(file_content)
                            content = content_bytes.decode("utf-8")
                        except Exception:
                            content = file_content
                    elif isinstance(file_content, bytes):
                        content = file_content.decode("utf-8")
                    else:
                        content = str(file_content)

                    json_data = json.loads(content)
                    formatted_json = json.dumps(json_data, indent=2)
                    return f"**📄 {file_name}**\n```json\n{formatted_json}\n```"
                except Exception:
                    return f"**📄 {file_name}** (JSON file)"
            elif file_type.lower() in ["txt", "text/plain"]:
                # For text files, show content preview
                try:
                    import base64

                    # Convert base64 content to string if needed
                    if isinstance(file_content, str):
                        try:
                            content_bytes = base64.b64decode(file_content)
                            content = content_bytes.decode("utf-8")
                        except Exception:
                            content = file_content
                    elif isinstance(file_content, bytes):
                        content = file_content.decode("utf-8")
                    else:
                        content = str(file_content)

                    # Show first 500 characters
                    preview = content[:500] + "..." if len(content) > 500 else content
                    return f"**📝 {file_name}**\n```\n{preview}\n```"
                except Exception:
                    return f"**📝 {file_name}** (Text file)"
            else:
                return f"**📁 {file_name}**"
        else:
            return match.group(0)

    processed_text = re.sub(file_pattern, replace_file_reference, text)
    return processed_text


def _display_answer_with_inline_files(content):
    """Display answer content with inline file displays."""
    import re

    parts = re.split(r"```file_display_([^`]+)```", content)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            file_key = part
            if file_key in st.session_state.get("file_displays", {}):
                file_info = st.session_state.file_displays[file_key]
                _display_single_file_inline(file_info, 0)
            else:
                st.markdown(f"[File not found: {file_key}]")


def _display_single_file_inline(file_info, i=0):
    """Display a single file inline based on its type."""

    file_name = file_info.get("name", f"file_{i}")
    file_content = file_info.get("content", "")
    file_type = file_info.get("type", "unknown")

    if "file_display_counter" not in st.session_state:
        st.session_state.file_display_counter = 0
    st.session_state.file_display_counter += 1

    unique_key = f"{file_name}_{st.session_state.file_display_counter}_{i}"

    try:
        if file_type.lower() in ["csv", "text/csv"]:
            _display_csv_file_inline(file_content, file_name, unique_key)
        elif file_type.lower() in ["png", "jpg", "jpeg", "gif", "bmp", "image/png", "image/jpeg", "image/gif"]:
            _display_image_file_inline(file_content, file_name, unique_key)
        elif file_type.lower() in ["json", "application/json"]:
            _display_json_file_inline(file_content, file_name, unique_key)
        elif file_type.lower() in ["txt", "text/plain"]:
            _display_text_file_inline(file_content, file_name, unique_key)
        else:
            st.download_button(
                label=f"📥 Download {file_name}",
                data=file_content if isinstance(file_content, bytes) else file_content.encode(),
                file_name=file_name,
                mime="application/octet-stream",
                key=f"download_{unique_key}",
            )
    except Exception as e:
        st.error(f"Error displaying file {file_name}: {e}")
        st.download_button(
            label=f"📥 Download {file_name}",
            data=file_content if isinstance(file_content, bytes) else file_content.encode(),
            file_name=file_name,
            mime="application/octet-stream",
            key=f"download_fallback_{unique_key}",
        )


def _display_csv_file_inline(content, file_name, unique_key=None):
    """Display CSV content as a table inline."""
    try:
        import io
        import time

        import pandas as pd

        if isinstance(content, str):
            import base64

            content_bytes = base64.b64decode(content)
            content = content_bytes.decode("utf-8")

        elif isinstance(content, bytes):
            content = content.decode("utf-8")

        df = pd.read_csv(io.StringIO(content))

        st.dataframe(df, use_container_width=True)

        csv_data = df.to_csv(index=False)
        csv_base64 = base64.b64encode(csv_data.encode("utf-8")).decode("utf-8")
        data_uri = f"data:text/csv;base64,{csv_base64}"

        unique_key = unique_key or f"{file_name}_{int(time.time() * 1000)}"
        st.markdown(f"[📥 Download {file_name}]({data_uri})", unsafe_allow_html=True, key=f"csv_download_{unique_key}")

    except Exception as e:
        st.error(f"Error parsing CSV {file_name}: {e}")
        # Fallback to text display
        st.code(content[:1000] + "..." if len(content) > 1000 else content)


def _display_image_file_inline(content, file_name, unique_key=None):
    """Display image content inline using data URI."""
    try:
        import base64

        if isinstance(content, str):
            if content.startswith("data:"):
                data_uri = content
            else:
                data_uri = f"data:image/png;base64,{content}"
        elif hasattr(content, "getvalue"):
            import io

            content_bytes = content.getvalue()
            base64_str = base64.b64encode(content_bytes).decode("utf-8")
            data_uri = f"data:image/png;base64,{base64_str}"
        elif isinstance(content, io.BytesIO):
            content_bytes = content.getvalue()
            base64_str = base64.b64encode(content_bytes).decode("utf-8")
            data_uri = f"data:image/png;base64,{base64_str}"
        elif isinstance(content, bytes):
            base64_str = base64.b64encode(content).decode("utf-8")
            data_uri = f"data:image/png;base64,{base64_str}"
        else:
            content_str = str(content)
            base64_str = base64.b64encode(content_str.encode("utf-8")).decode("utf-8")
            data_uri = f"data:image/png;base64,{base64_str}"

        st.markdown(f"![{file_name}]({data_uri})", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error displaying image {file_name}: {e}")


def _display_json_file_inline(content, file_name, unique_key=None):
    """Display JSON content inline."""
    try:
        import json

        if isinstance(content, bytes):
            content = content.decode("utf-8")

        json_data = json.loads(content)
        st.json(json_data)

    except Exception as e:
        st.error(f"Error parsing JSON {file_name}: {e}")
        st.code(content[:1000] + "..." if len(content) > 1000 else content)


def _display_text_file_inline(content, file_name, unique_key=None):
    """Display text content inline."""
    try:

        if isinstance(content, bytes):
            content = content.decode("utf-8")

        st.code(content[:2000] + "..." if len(content) > 2000 else content)

    except Exception as e:
        st.error(f"Error displaying text {file_name}: {e}")


def create_agent(streaming_enabled: bool = False):
    """Create and configure the data analyst agent with E2B tools."""
    tool = E2BInterpreterTool(name="data-analyzer-e2b", connection=E2B())

    llm = AnthropicLLM(
        name="claude",
        model="claude-3-7-sonnet-20250219",
        temperature=0.1,
        max_tokens=8000,
        budget_tokens=4000,
    )

    streaming_config = StreamingConfig(enabled=streaming_enabled, mode=StreamingMode.ALL)

    agent = Agent(
        name="DataAnalyst",
        id="DataAnalyst",
        llm=llm,
        tools=[tool],
        role=AGENT_ROLE,
        max_loops=20,
        inference_mode=InferenceMode.FUNCTION_CALLING,
        streaming=streaming_config,
    )

    return agent


def analyze_csv_file(
    file_obj: io.BytesIO, analysis_prompt: str = None, streaming_enabled: bool = False
) -> tuple[str, dict]:
    """
    Analyze a CSV file using the data analyst agent.

    Args:
        file_obj: BytesIO object containing the CSV or text file
        analysis_prompt: Optional specific analysis prompt
        streaming_enabled: Whether to enable streaming

    Returns:
        Tuple of (analysis_result, generated_files)
    """
    try:
        if "loading_placeholder" not in st.session_state:
            st.session_state.loading_placeholder = None
        if "is_loading" not in st.session_state:
            st.session_state.is_loading = False

        if streaming_enabled:
            _show_loading_spinner()
            st.session_state.is_loading = True

        agent = create_agent(streaming_enabled=streaming_enabled)

        if not analysis_prompt:
            analysis_prompt = """
            Analyze the uploaded CSV file and provide a comprehensive analysis:

            1. **Data Overview**: Load the data and provide basic statistics, shape, and column information

            2. **Data Quality Assessment**: Check for missing values, duplicates, and data types

            3. **Exploratory Data Analysis**:
               - Generate multiple visualizations (histograms, scatter plots, correlation heatmaps, box plots, etc.)
               - Save each plot as a PNG file with descriptive names
               - Reference saved images in your response using [filename.png] format

            4. **Statistical Analysis**: Perform relevant statistical tests and calculations

            5. **Key Insights and Patterns**: Identify trends, correlations, and interesting findings

            6. **Visualizations**: Create at least 3-5 different types of charts/plots and save them as files

            7. **Recommendations**: Provide actionable insights based on your analysis

            IMPORTANT: Always save your visualizations as PNG files and reference them
             in your response using [filename.png] format so they display inline.
            """

        analysis_prompt = "return in the ouput just ![sample_data.csv]"
        result = agent.run(
            input_data={"input": analysis_prompt, "files": [file_obj]},
        )

        content = result.output.get("content", "")
        files = result.output.get("files", {})

        if files and isinstance(files, dict):
            temp_generated_files = []
            for file_name, file_content in files.items():
                file_info = {"name": file_name, "content": file_content, "type": "unknown"}
                temp_generated_files.append(file_info)

            original_files = st.session_state.get("generated_files", [])
            st.session_state.generated_files = temp_generated_files

            try:
                processed_content = _process_file_references_in_text(content)
            finally:
                st.session_state.generated_files = original_files

            return processed_content, files
        else:
            return content, files

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return f"Error during analysis: {str(e)}", {}


def run_agent_with_streaming(
    file_obj: io.BytesIO, analysis_prompt: str, send_handler: AsyncStreamingIteratorCallbackHandler
) -> tuple[str, dict]:
    """
    Run the data analyst agent with streaming support.

    Args:
        file_obj: BytesIO object containing the CSV or text file
        analysis_prompt: Analysis prompt
        send_handler: Streaming callback handler

    Returns:
        Tuple of (analysis_result, generated_files)
    """
    try:
        agent = create_agent(streaming_enabled=True)

        if not analysis_prompt:
            analysis_prompt = """
            Analyze the uploaded CSV file and provide a comprehensive analysis:

            1. **Data Overview**: Load the data and provide basic statistics, shape, and column information

            2. **Data Quality Assessment**: Check for missing values, duplicates, and data types

            3. **Exploratory Data Analysis**:
               - Generate multiple visualizations (histograms, scatter plots, correlation heatmaps, box plots, etc.)
               - Save each plot as a PNG file with descriptive names
               - Reference saved images in your response using [filename.png] format

            4. **Statistical Analysis**: Perform relevant statistical tests and calculations

            5. **Key Insights and Patterns**: Identify trends, correlations, and interesting findings

            6. **Visualizations**: Create at least 3-5 different types of charts/plots and save them as files

            7. **Recommendations**: Provide actionable insights based on your analysis

            IMPORTANT: Always save your visualizations as PNG files and reference them
             in your response using [filename.png] format so they display inline.
            """

        flow = Workflow(
            flow=Flow(nodes=[agent]),
        )

        result = flow.run(
            input_data={"input": analysis_prompt, "files": [file_obj]}, config=RunnableConfig(callbacks=[send_handler])
        )

        content = result.output[agent.id]["output"].get("content", "")
        files = result.output[agent.id]["output"].get("files", {})

        return content, files

    except Exception as e:
        logger.error(f"Streaming analysis failed: {e}")
        return f"Error during analysis: {str(e)}", {}


def run_agent_with_streaming_sync(file_obj: io.BytesIO, analysis_prompt: str) -> tuple[str, dict]:
    """

    The use_column_width parameter has been deprecated and will be removed in a future release.
     Please utilize the use_container_width parameter instead
    Error displaying image department_distribution.png: cannot identify image file <_io.BytesIO object at 0x2871c3510>
    Run the data analyst agent with REAL streaming support using AsyncStreamingIteratorCallbackHandler.

    Args:
        file_obj: BytesIO object containing the CSV or text file
        analysis_prompt: Analysis prompt

    Returns:
        Tuple of (analysis_result, generated_files)
    """
    import asyncio

    # Initialize session state for loading
    if "loading_placeholder" not in st.session_state:
        st.session_state.loading_placeholder = None
    if "is_loading" not in st.session_state:
        st.session_state.is_loading = False

    # Show loading spinner when analysis starts
    _show_loading_spinner()
    st.session_state.is_loading = True

    # Run the async function directly
    try:
        result = asyncio.run(run_agent_async_streaming(file_obj, analysis_prompt))
        return result
    except Exception as e:
        logger.error(f"Streaming analysis failed: {e}")
        return f"Error during analysis: {str(e)}", {}
    finally:
        # Ensure loading spinner is hidden even if there's an error
        if st.session_state.is_loading:
            _hide_loading_spinner()
            st.session_state.is_loading = False


def run_agent_sync(
    file_obj: io.BytesIO, analysis_prompt: str, send_handler: AsyncStreamingIteratorCallbackHandler
) -> tuple[str, dict]:
    """
    Run the data analyst agent with streaming support (sync version for executor).

    Args:
        file_obj: BytesIO object containing the CSV or text file
        analysis_prompt: Analysis prompt
        send_handler: Streaming callback handler

    Returns:
        Tuple of (analysis_result, generated_files)
    """
    try:
        agent = create_agent(streaming_enabled=True)

        # Set default prompt if none provided
        if not analysis_prompt:
            analysis_prompt = """
            Analyze the uploaded CSV file and provide a comprehensive analysis:

            1. **Data Overview**: Load the data and provide basic statistics, shape, and column information

            2. **Data Quality Assessment**: Check for missing values, duplicates, and data types

            3. **Exploratory Data Analysis**:
               - Generate multiple visualizations (histograms, scatter plots, correlation heatmaps, box plots, etc.)
               - Save each plot as a PNG file with descriptive names
               - Reference saved images in your response using [filename.png] format

            4. **Statistical Analysis**: Perform relevant statistical tests and calculations

            5. **Key Insights and Patterns**: Identify trends, correlations, and interesting findings

            6. **Visualizations**: Create at least 3-5 different types of charts/plots and save them as files

            7. **Recommendations**: Provide actionable insights based on your analysis

            IMPORTANT: Always save your visualizations as PNG files and reference them
            in your response using [filename.png] format so they display inline.
            """

        flow = Workflow(
            flow=Flow(nodes=[agent]),
        )

        result = flow.run(
            input_data={"input": analysis_prompt, "files": [file_obj]}, config=RunnableConfig(callbacks=[send_handler])
        )

        content = result.output[agent.id]["output"].get("content", "")
        files = result.output[agent.id]["output"].get("files", {})

        return content, files

    except Exception as e:
        logger.error(f"Streaming analysis failed: {e}")
        return f"Error during analysis: {str(e)}", {}


async def run_agent_async_streaming(file_obj: io.BytesIO, analysis_prompt: str) -> tuple[str, dict]:
    """
    Run the data analyst agent with REAL async streaming support.

    Args:
        file_obj: BytesIO object containing the CSV or text file
        analysis_prompt: Analysis prompt

    Returns:
        Tuple of (analysis_result, generated_files)
    """
    try:
        # Initialize streaming UI
        init_stream_ui(reset=True)

        # Create streaming handler
        send_handler = AsyncStreamingIteratorCallbackHandler()

        # Get current event loop
        current_loop = asyncio.get_running_loop()

        # Create streaming task
        streaming_task = current_loop.create_task(_send_stream_events_by_ws(send_handler))
        await asyncio.sleep(0.01)

        # Run agent in executor (like the working example)
        response, files = await current_loop.run_in_executor(
            None, run_agent_sync, file_obj, analysis_prompt, send_handler
        )

        # Files are now captured from E2B tool streaming output, not from agent final output

        # Wait for streaming to complete
        await streaming_task

        # Display all generated files at the end
        _display_all_files_at_end()

        # Process the response to replace file placeholders with actual file content
        processed_response = _process_file_references_in_text(response)

        return processed_response, files

    except Exception as e:
        logger.error(f"Async streaming analysis failed: {e}")
        return f"Error during analysis: {str(e)}", {}


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

        # Handle different delta formats
        if isinstance(delta, str):
            # Delta is just a string (final answer)
            # Only call callback if this is new content
            if "last_answer_content" not in st.session_state or st.session_state.last_answer_content != delta:
                st.session_state.last_answer_content = delta
                streamlit_callback("answer", delta)
        elif isinstance(delta, dict):
            # Delta is a dictionary with step and content
            step = delta.get("step", "")
            content = delta.get("content", "")

            if step and content:
                streamlit_callback(step, content)
            elif content:
                # Content without step, treat as answer
                # Only call callback if this is new content
                if "last_answer_content" not in st.session_state or st.session_state.last_answer_content != content:
                    st.session_state.last_answer_content = content
                    streamlit_callback("answer", content)


def save_analysis_files(files: dict | list, output_dir: str = "./analysis_outputs") -> list[str]:
    """
    Save files generated during analysis.

    Args:
        files: Dictionary of generated files or list of BytesIO objects
        output_dir: Directory to save files

    Returns:
        List of saved file paths
    """
    if not files:
        return []

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    # Handle list of BytesIO objects (new format)
    if isinstance(files, list):
        for file_bytesio in files:
            try:
                # Get file metadata from BytesIO object
                file_name = getattr(file_bytesio, "name", f"file_{id(file_bytesio)}.bin")

                # Read content from BytesIO
                file_data = file_bytesio.read()
                file_bytesio.seek(0)  # Reset position for potential future reads

                output_path = os.path.join(output_dir, file_name)

                with open(output_path, "wb") as f:
                    f.write(file_data)

                saved_files.append(output_path)
                logger.info(f"Saved analysis file: {output_path}")

            except Exception as e:
                logger.error(f"Failed to save file {getattr(file_bytesio, 'name', 'unknown')}: {e}")

    # Handle dictionary format (legacy)
    elif isinstance(files, dict):
        for file_path, file_content in files.items():
            try:
                # Handle base64 encoded content
                if isinstance(file_content, str) and file_content.startswith("data:"):
                    import base64

                    header, base64_content = file_content.split(",", 1)
                    file_data = base64.b64decode(base64_content)
                else:
                    file_data = file_content.encode() if isinstance(file_content, str) else file_content

                output_path = os.path.join(output_dir, os.path.basename(file_path))

                with open(output_path, "wb") as f:
                    f.write(file_data)

                saved_files.append(output_path)
                logger.info(f"Saved analysis file: {output_path}")

            except Exception as e:
                logger.error(f"Failed to save file {file_path}: {e}")

    return saved_files
