import asyncio
import io
import os
from typing import List

from dynamiq import Workflow
from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.connections import E2B
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms import Anthropic as AnthropicLLM
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from dynamiq.utils.logger import logger
import streamlit as st

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
    import time
    
    # Initialize loading state if not exists
    if "loading_placeholder" not in st.session_state:
        st.session_state.loading_placeholder = None
    if "is_loading" not in st.session_state:
        st.session_state.is_loading = False
    
    if st.session_state.loading_placeholder is None:
        st.session_state.loading_placeholder = st.empty()
    
    # Create a more sophisticated loading animation
    with st.session_state.loading_placeholder.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("""
            <div style="display: flex; align-items: center; justify-content: center; height: 40px;">
                <div style="width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid #3498db; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("🤖 **Agent is analyzing your data...**")
            st.markdown("*This may take a few moments*")
    
    # Add CSS for spinning animation
    st.markdown("""
    <style>
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)

def _hide_loading_spinner():
    """Hide the loading spinner."""
    import streamlit as st
    
    # Initialize loading state if not exists
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
        # Decode base64 to bytes
        content_bytes = base64.b64decode(base64_string)
        
        # Create BytesIO object
        bytesio_obj = io.BytesIO(content_bytes)
        bytesio_obj.name = file_name
        bytesio_obj.content_type = content_type
        
        return bytesio_obj
    except Exception as e:
        print(f"DEBUG: Error converting base64 to BytesIO: {e}")
        return None


def _is_valid_base64(s):
    """Check if a string is valid base64."""
    import base64
    try:
        if isinstance(s, str):
            # Try to decode and re-encode to check validity
            decoded = base64.b64decode(s, validate=True)
            # Check if it's reasonable size (not too small or too large)
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
        try:
            print(f"DEBUG: Processing file {i}: {type(file_obj)}")
            print(f"DEBUG: File {i} attributes: {dir(file_obj) if hasattr(file_obj, '__dict__') else 'No attributes'}")
            
            # Check if it's a SerializationIterator object
            if hasattr(file_obj, 'iterator') and hasattr(file_obj, 'index'):
                print(f"DEBUG: File {i} is SerializationIterator, extracting BytesIO from iterator")
                # Extract the BytesIO object from the iterator
                bytesio_obj = file_obj.iterator
                if hasattr(bytesio_obj, 'getvalue'):
                    file_name = getattr(bytesio_obj, 'name', f'file_{i}')
                    file_content = bytesio_obj.getvalue()
                    file_type = getattr(bytesio_obj, 'content_type', 'unknown')
                    
                    # Convert to base64 for storage
                    import base64
                    base64_content = base64.b64encode(file_content).decode('utf-8')
                    
                    file_info = {
                        "name": file_name,
                        "content": base64_content,
                        "type": file_type,
                        "original_size": len(file_content)
                    }
                    
                    print(f"DEBUG: Processed SerializationIterator file {i}: {file_name} ({file_type}) - {len(file_content)} bytes")
                    processed_files.append(file_info)
                else:
                    print(f"DEBUG: SerializationIterator {i} does not contain BytesIO object")
                    
            elif isinstance(file_obj, dict) and "content" in file_obj and _is_valid_base64(file_obj["content"]):
                # This is a file info dict with base64 content and metadata from serialization
                print(f"DEBUG: File {i} is file info dict with base64 content")
                file_name = file_obj.get("name", f"file_{i}")
                content_type = file_obj.get("content_type", "unknown")
                base64_content = file_obj["content"]
                description = file_obj.get("description", "")
                original_size = file_obj.get("original_size", 0)
                
                # Use base64 content directly without BytesIO conversion
                file_info = {
                    "name": file_name,
                    "content": base64_content,
                    "type": content_type,
                    "description": description,
                    "original_size": original_size
                }
                
                print(f"DEBUG: Processed file info dict {i}: {file_name} ({content_type}) - {original_size} bytes")
                processed_files.append(file_info)
                    
            elif isinstance(file_obj, str) and _is_valid_base64(file_obj):
                # This is a legacy base64 string from serialization (fallback)
                print(f"DEBUG: File {i} is legacy base64 string, converting to BytesIO")
                # Try to extract file info from context or use defaults
                file_name = f"file_{i}"
                content_type = "unknown"
                
                # Convert base64 to BytesIO
                bytesio_obj = _convert_base64_to_bytesio(file_obj, file_name, content_type)
                if bytesio_obj:
                    file_content = bytesio_obj.getvalue()
                    base64_content = file_obj  # Already base64
                    
                    file_info = {
                        "name": file_name,
                        "content": base64_content,
                        "type": content_type,
                        "original_size": len(file_content)
                    }
                    
                    print(f"DEBUG: Converted legacy base64 file {i}: {file_name} - {len(file_content)} bytes")
                    processed_files.append(file_info)
                else:
                    print(f"DEBUG: Failed to convert legacy base64 file {i}")
                    
            elif hasattr(file_obj, 'name') and hasattr(file_obj, 'getvalue'):
                # This is a BytesIO object from E2B
                file_name = getattr(file_obj, 'name', f'file_{i}')
                file_content = file_obj.getvalue()
                file_type = getattr(file_obj, 'content_type', 'unknown')
                
                # Convert to base64 for storage
                import base64
                base64_content = base64.b64encode(file_content).decode('utf-8')
                
                file_info = {
                    "name": file_name,
                    "content": base64_content,
                    "type": file_type,
                    "original_size": len(file_content)
                }
                
                print(f"DEBUG: Processed BytesIO file {i}: {file_name} ({file_type}) - {len(file_content)} bytes")
                processed_files.append(file_info)
            elif isinstance(file_obj, str):
                # This is a regular string (not base64)
                print(f"DEBUG: File {i} is regular string (not base64): {file_obj[:100] if len(str(file_obj)) > 100 else str(file_obj)}...")
                # Skip non-base64 strings
                continue
            else:
                print(f"DEBUG: File {i} is unknown type: {type(file_obj)}")
                print(f"DEBUG: File {i} attributes: {dir(file_obj) if hasattr(file_obj, '__dict__') else 'No attributes'}")
                
        except Exception as e:
            print(f"DEBUG: Error processing file {i}: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
    
    return processed_files

def streamlit_callback(step: str, content):
    """Callback function to handle streaming content display with tool input and file display support."""
    if not content:
        return

    if step not in ["reasoning", "answer", "tool"]:
        return

    # Initialize streaming UI state if not exists
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
        # Ensure it's always a list, not a dict
        st.session_state.generated_files = []
    if "loading_placeholder" not in st.session_state:
        st.session_state.loading_placeholder = None
    if "is_loading" not in st.session_state:
        st.session_state.is_loading = False

    # Always use "reasoning" as the step key to keep everything in the same block
    step_key = "reasoning"
    display_name = "Reasoning"

    # Handle the reasoning step
    if step == "reasoning" and isinstance(content, dict):
        # Hide loading spinner when reasoning starts
        if st.session_state.is_loading:
            _hide_loading_spinner()
            st.session_state.is_loading = False
        
        thought = content.get("thought", "")
        loop_num = content.get("loop_num", 0)

        current_loop = st.session_state.current_loop_nums.get(step_key, -1)
        if current_loop != loop_num and current_loop != -1:
            # Add separator when loop number changes
            if step_key in st.session_state.step_contents:
                st.session_state.step_contents[step_key] += "\n\n"

        st.session_state.current_loop_nums[step_key] = loop_num
        content_to_display = thought
    
    # Handle the tool step - ONLY process E2B tool calls, hide all others
    elif step == "tool" and isinstance(content, dict):
        # Hide loading spinner when tool execution starts
        if st.session_state.is_loading:
            _hide_loading_spinner()
            st.session_state.is_loading = False

        tool_name = content.get("name", "Unknown Tool")
        tool_input = content.get("input", "")
        tool_result = content.get("result", "")
        tool_files = content.get("files", [])
        
        print(f"DEBUG: Tool step - Tool name: {tool_name}")
        print(f"DEBUG: Tool step - Tool input: {tool_input}")
        print(f"DEBUG: Tool step - Tool result: {tool_result}")
        print(f"DEBUG: Tool step - Tool files: {tool_files}")
        print(f"DEBUG: Tool step - Tool files type: {type(tool_files)}")
        print(f"DEBUG: Tool step - Tool files length: {len(tool_files) if isinstance(tool_files, list) else 'Not a list'}")
        print(f"File: type{ type(tool_files[0]) if tool_files else "none"}")
        print(f"File: type{ str(tool_files[0])[:100] if tool_files and len(str(tool_files[0])) > 100 else str(tool_files[0]) if tool_files else "none"}")

        # Debug: Show all keys in the content dict
        print(f"DEBUG: All content keys: {list(content.keys()) if isinstance(content, dict) else 'Not a dict'}")
        for key, value in content.items():
            value_str = str(value)
            print(f"DEBUG:   {key}: {type(value)} - {value_str[:100] if len(value_str) > 100 else value_str}...")
        
        # Only process E2B tool calls - completely hide all other tools
        if "e2b" in tool_name.lower() or "interpreter" in tool_name.lower():
            content_to_display = ""
            
            # Extract Python code from tool input
            if tool_input:
                # Handle both string and dict inputs
                if isinstance(tool_input, str):
                    # If it's a string, it might be JSON that needs parsing
                    try:
                        import json
                        tool_input_dict = json.loads(tool_input)
                        python_code = tool_input_dict.get("python", "")
                    except (json.JSONDecodeError, AttributeError):
                        # If not JSON, check if it contains Python code patterns
                        python_code = tool_input if "import " in tool_input or "print(" in tool_input else ""
                elif isinstance(tool_input, dict):
                    python_code = tool_input.get("python", "")
                else:
                    python_code = ""
                
                if python_code:
                    content_to_display += f"\n\n**🐍 Generated Python Code:**\n\n"
                    content_to_display += f"```python\n{python_code}\n```\n\n"
            
            # Process files from tool_files field using the new conversion function
            print(f"DEBUG: Processing tool_files: {tool_files}")
            print(f"DEBUG: tool_files is list: {isinstance(tool_files, list)}")
            print(f"DEBUG: tool_files is truthy: {bool(tool_files)}")
            
            # Use the new conversion function to handle both BytesIO and base64 strings
            processed_files = _process_tool_files_with_conversion(tool_files)
            
            if processed_files:
                print(f"DEBUG: Successfully processed {len(processed_files)} files")
                for file_info in processed_files:
                    print(f"DEBUG: Adding processed file: {file_info['name']} ({file_info['type']}) - {file_info.get('original_size', 0)} bytes")
                    st.session_state.generated_files.append(file_info)
                print(f"DEBUG: Added {len(processed_files)} files to generated_files. Total count: {len(st.session_state.generated_files)}")
            else:
                print(f"DEBUG: No files processed from tool_files")
                print(f"DEBUG: tool_files value: {tool_files}")
                print(f"DEBUG: tool_files type: {type(tool_files)}")
        else:
            # For non-E2B tools, return empty content to hide them completely
            return
    
    # Handle the answer step - create separate answer block with inline file displays
    elif step == "answer":
        # Hide loading spinner when answer starts
        if st.session_state.is_loading:
            _hide_loading_spinner()
            st.session_state.is_loading = False
        
        step_key = "answer"
        display_name = "Answer"
        content_to_display = str(content) if not isinstance(content, str) else content
        
        # Files are now captured from E2B tool streaming output, not from answer step
        # Debug: Show current generated files
        generated_files = st.session_state.get("generated_files", [])
        print(f"DEBUG: Answer step - Current generated files count: {len(generated_files)}")
        for i, f in enumerate(generated_files):
            print(f"  {i+1}. {f.get('name', 'unknown')} ({f.get('type', 'unknown')}) - {len(str(f.get('content', '')))} chars")
        
        # Debug: Show answer content for file references
        print(f"DEBUG: Answer content length: {len(content_to_display)}")
        print(f"DEBUG: Answer content preview: {content_to_display[:200] if len(content_to_display) > 200 else content_to_display}...")
        
        # Check for file references in the answer
        import re
        file_refs = re.findall(r'\[([^\]]+\.(?:png|jpg|jpeg|gif|bmp|svg|csv|json|txt))\]', content_to_display)
        print(f"DEBUG: Found {len(file_refs)} file references in answer: {file_refs}")
        
        # Debug: Show file_displays state
        file_displays = st.session_state.get("file_displays", {})
        print(f"DEBUG: File displays count: {len(file_displays)}")
        for key, info in file_displays.items():
            print(f"  {key}: {info.get('name', 'unknown')} ({info.get('type', 'unknown')})")
    else:
        content_to_display = str(content) if not isinstance(content, str) else content

    # Create a new box for a step on first occurrence
    if step_key not in st.session_state.step_placeholders:
        if step_key == "reasoning":
            # Use expander for reasoning block
            with st.expander("🤔 Reasoning", expanded=False):
                placeholder = st.empty()
        else:
            box = st.container(border=True)
            with box:
                st.markdown(f"### {display_name}")
                if step_key == "answer":
                    # For answer step, create a placeholder for the entire answer content
                    placeholder = st.empty()
                else:
                    placeholder = st.empty()
        st.session_state.step_placeholders[step_key] = placeholder
        st.session_state.step_contents[step_key] = ""
        st.session_state.step_order.append(step_key)

    # Append the incoming content and update the box
    st.session_state.step_contents[step_key] += content_to_display
    
    # For answer step, just display the content without inline files
    if step_key == "answer":
        # Process file references in the accumulated content
        processed_content = _process_file_references_in_text(st.session_state.step_contents[step_key])
        # Clear the placeholder and display the processed content
        with st.session_state.step_placeholders[step_key].container():
            _display_answer_with_inline_files(processed_content)
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
        file_type = file_info.get('type', 'unknown')
        original_size = file_info.get('original_size', 0)
        description = file_info.get('description', '')
        
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
                    _display_csv_file_inline(file_info.get('content', ''), file_info.get('name', f'file_{i}'), f"end_{i}")
                elif file_type.lower() in ["png", "jpg", "jpeg", "gif", "bmp", "image/png", "image/jpeg", "image/gif"]:
                    _display_image_file_inline(file_info.get('content', ''), file_info.get('name', f'file_{i}'), f"end_{i}")
                elif file_type.lower() in ["json", "application/json"]:
                    _display_json_file_inline(file_info.get('content', ''), file_info.get('name', f'file_{i}'), f"end_{i}")
                elif file_type.lower() in ["txt", "text/plain"]:
                    _display_text_file_inline(file_info.get('content', ''), file_info.get('name', f'file_{i}'), f"end_{i}")
                else:
                    # Default: show as downloadable file
                    _display_single_file_inline(file_info, i)
            except Exception as e:
                st.error(f"Error displaying file {file_info.get('name', f'file_{i}')}: {e}")
        
        st.markdown("---")


def _process_file_references_in_text(text):
    """Process file references in text and replace with actual file displays."""
    import re
    
    print(f"DEBUG: Processing text for file references: {text[:200] if len(text) > 200 else text}...")
    
    # Pattern to match file references like [filename.ext] or [filename]
    file_pattern = r'\[([^\]]+\.(?:csv|png|jpg|jpeg|gif|bmp|json|txt|pdf|html|xml|yaml|yml))\](?!\])'
    
    # Find all file references in the text
    file_refs = re.findall(file_pattern, text)
    print(f"DEBUG: Found {len(file_refs)} file references in text: {file_refs}")
    
    def replace_file_reference(match):
        file_name = match.group(1)
        print(f"DEBUG: Processing file reference: '{file_name}'")
        
        # Find the file in generated files
        file_info = None
        generated_files = st.session_state.get("generated_files", [])
        print(f"DEBUG: Searching in {len(generated_files)} generated files")
        
        for f in generated_files:
            stored_name = f.get("name", "")
            print(f"DEBUG: Checking stored file '{stored_name}' against '{file_name}'")
            
            # Try exact match first, then endswith, then contains
            exact_match = stored_name == file_name
            endswith_match = stored_name.endswith(file_name)
            contains_match = file_name in stored_name
            path_match_forward = stored_name.split('/')[-1] == file_name
            path_match_backward = stored_name.split('\\')[-1] == file_name
            
            print(f"DEBUG:   Exact: {exact_match}, Endswith: {endswith_match}, Contains: {contains_match}, Path(/): {path_match_forward}, Path(\\): {path_match_backward}")
            
            if (exact_match or endswith_match or contains_match or path_match_forward or path_match_backward):
                file_info = f
                print(f"DEBUG: ✅ MATCH FOUND! File: {stored_name}")
                break
            else:
                print(f"DEBUG: ❌ No match for '{stored_name}'")
        
        if file_info:
            # Create a unique key for this file display
            file_key = f"file_display_{hash(file_name)}_{len(generated_files)}"
            print(f"DEBUG: Creating file key: {file_key}")
            
            # Store the file info for later display
            if "file_displays" not in st.session_state:
                st.session_state.file_displays = {}
            st.session_state.file_displays[file_key] = file_info
            print(f"DEBUG: Stored file info in file_displays with key: {file_key}")
            
            # Return a placeholder that will be replaced with actual content
            placeholder = f"```{file_key}```"
            print(f"DEBUG: Returning placeholder: {placeholder}")
            return placeholder
        else:
            # File not found, return original reference
            print(f"DEBUG: ❌ File '{file_name}' not found in generated files")
            print(f"DEBUG: Returning original reference: {match.group(0)}")
            return match.group(0)
    
    # Replace file references with placeholders
    processed_text = re.sub(file_pattern, replace_file_reference, text)
    
    # Split text by file placeholders and process each part
    parts = re.split(r'```([^`]+)```', processed_text)
    
    result_parts = []
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Regular text
            result_parts.append(part)
        else:
            # File placeholder
            file_key = part
            if file_key in st.session_state.get("file_displays", {}):
                file_info = st.session_state.file_displays[file_key]
                # Create a container for the file display
                result_parts.append(f"\n\n**📁 {file_info.get('name', 'File')}**\n\n")
                result_parts.append(f"```file_display_{file_key}```\n\n")
            else:
                result_parts.append(f"[{file_key}]")
    
    final_result = "".join(result_parts)
    return final_result


def _display_answer_with_inline_files(content):
    """Display answer content with inline file displays."""
    import re
    
    print(f"DEBUG: _display_answer_with_inline_files called with content length: {len(content)}")
    print(f"DEBUG: Content preview: {content[:200] if len(content) > 200 else content}...")
    
    # Split content by file display placeholders
    parts = re.split(r'```file_display_([^`]+)```', content)
    print(f"DEBUG: Split content into {len(parts)} parts")
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Regular text content
            print(f"DEBUG: Part {i} (text): {part[:50] if len(part) > 50 else part}...")
            if part.strip():
                st.markdown(part)
        else:
            # File display placeholder
            file_key = part
            print(f"DEBUG: Part {i} (file key): {file_key}")
            if file_key in st.session_state.get("file_displays", {}):
                file_info = st.session_state.file_displays[file_key]
                print(f"DEBUG: Found file info for key {file_key}: {file_info.get('name', 'unknown')}")
                _display_single_file_inline(file_info, 0)
            else:
                print(f"DEBUG: File key {file_key} not found in file_displays!")
                st.markdown(f"[File not found: {file_key}]")


def _detect_mime_type_from_filename(filename):
    """Detect MIME type from file extension."""
    extension = filename.lower().split(".")[-1] if "." in filename else ""
    
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg", 
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "ico": "image/x-icon",
        "svg": "image/svg+xml",
        "pdf": "application/pdf",
        "txt": "text/plain",
        "csv": "text/csv",
        "json": "application/json",
        "html": "text/html",
        "css": "text/css",
        "js": "application/javascript",
        "md": "text/markdown",
    }
    
    return mime_map.get(extension, "image/png")  # Default to PNG for images


def _display_single_file_inline(file_info, i=0):
    """Display a single file inline based on its type."""
    import time
    import hashlib
    import random
    import uuid
    
    file_name = file_info.get("name", f"file_{i}")
    file_content = file_info.get("content", "")
    file_type = file_info.get("type", "unknown")
    
    # Create a more unique key using file name, content hash, timestamp, random, UUID, and session counter
    content_hash = hashlib.md5(str(file_content).encode()).hexdigest()[:8]
    random_component = random.randint(1000, 9999)
    uuid_component = str(uuid.uuid4())[:8]
    
    # Use session state counter for additional uniqueness
    if "file_display_counter" not in st.session_state:
        st.session_state.file_display_counter = 0
    st.session_state.file_display_counter += 1
    
    unique_key = f"{file_name}_{content_hash}_{int(time.time() * 1000)}_{random_component}_{uuid_component}_{st.session_state.file_display_counter}_{i}"
    
    # Ensure the key is truly unique by checking against existing keys
    if "used_keys" not in st.session_state:
        st.session_state.used_keys = set()
    
    original_key = unique_key
    counter = 0
    while unique_key in st.session_state.used_keys:
        counter += 1
        unique_key = f"{original_key}_dup_{counter}"
    
    st.session_state.used_keys.add(unique_key)
    
    print(f"DEBUG: _display_single_file_inline called for '{file_name}' with unique key: {unique_key}")
    print(f"DEBUG: File type: '{file_type}'")
    print(f"DEBUG: Content type: {type(file_content)}")
    print(f"DEBUG: Content length: {len(str(file_content))}")
    
    try:
        if file_type.lower() in ["csv", "text/csv"]:
            print(f"DEBUG: Routing to CSV display for '{file_name}'")
            _display_csv_file_inline(file_content, file_name, unique_key)
        elif file_type.lower() in ["png", "jpg", "jpeg", "gif", "bmp", "image/png", "image/jpeg", "image/gif"]:
            print(f"DEBUG: Routing to IMAGE display for '{file_name}' (type: {file_type})")
            _display_image_file_inline(file_content, file_name, unique_key)
        elif file_type.lower() in ["json", "application/json"]:
            print(f"DEBUG: Routing to JSON display for '{file_name}'")
            _display_json_file_inline(file_content, file_name, unique_key)
        elif file_type.lower() in ["txt", "text/plain"]:
            print(f"DEBUG: Routing to TEXT display for '{file_name}'")
            _display_text_file_inline(file_content, file_name, unique_key)
        else:
            print(f"DEBUG: Unknown file type '{file_type}', showing as downloadable file")
            # Default: show as downloadable file
            st.download_button(
                label=f"📥 Download {file_name}",
                data=file_content if isinstance(file_content, bytes) else file_content.encode(),
                file_name=file_name,
                mime="application/octet-stream",
                key=f"download_{unique_key}"
            )
    except Exception as e:
        print(f"DEBUG: ❌ Error in _display_single_file_inline for '{file_name}': {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        st.error(f"Error displaying file {file_name}: {e}")
        # Fallback to download button
        st.download_button(
            label=f"📥 Download {file_name}",
            data=file_content if isinstance(file_content, bytes) else file_content.encode(),
            file_name=file_name,
            mime="application/octet-stream",
            key=f"download_fallback_{unique_key}"
        )


def _display_csv_file_inline(content, file_name, unique_key=None):
    """Display CSV content as a table inline."""
    try:
        import pandas as pd
        import io
        import time
        
        # Convert base64 content to string if needed
        if isinstance(content, str):
            try:
                # Try to decode as base64 first
                import base64
                content_bytes = base64.b64decode(content)
                content = content_bytes.decode('utf-8')
            except Exception:
                # If not base64, use as string
                pass
        elif isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Read CSV from string
        df = pd.read_csv(io.StringIO(content))
        
        # Display the dataframe
        st.dataframe(df, use_container_width=True)
        
        # Create data URI for download
        csv_data = df.to_csv(index=False)
        csv_base64 = base64.b64encode(csv_data.encode('utf-8')).decode('utf-8')
        data_uri = f"data:text/csv;base64,{csv_base64}"
        
        # Display as markdown link for download with unique key
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
        
        print(f"DEBUG: _display_image_file_inline - content type: {type(content)}")
        print(f"DEBUG: _display_image_file_inline - content length: {len(str(content)) if hasattr(content, '__len__') else 'No length'}")
        
        # Get base64 string
        if isinstance(content, str):
            if content.startswith("data:"):
                # Already a data URI
                data_uri = content
            else:
                # Plain base64 string, create data URI
                data_uri = f"data:image/png;base64,{content}"
        elif hasattr(content, 'getvalue'):
            # BytesIO object, convert to base64
            import io
            content_bytes = content.getvalue()
            base64_str = base64.b64encode(content_bytes).decode('utf-8')
            data_uri = f"data:image/png;base64,{base64_str}"
        elif isinstance(content, io.BytesIO):
            # BytesIO object, convert to base64
            content_bytes = content.getvalue()
            base64_str = base64.b64encode(content_bytes).decode('utf-8')
            data_uri = f"data:image/png;base64,{base64_str}"
        elif isinstance(content, bytes):
            # Raw bytes, convert to base64
            base64_str = base64.b64encode(content).decode('utf-8')
            data_uri = f"data:image/png;base64,{base64_str}"
        else:
            # Try to convert to string then to base64
            content_str = str(content)
            base64_str = base64.b64encode(content_str.encode('utf-8')).decode('utf-8')
            data_uri = f"data:image/png;base64,{base64_str}"
        
        print(f"DEBUG: _display_image_file_inline - data URI length: {len(data_uri)}")
        
        # Display the image using markdown with data URI
        st.markdown(f"![{file_name}]({data_uri})", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying image {file_name}: {e}")
        print(f"DEBUG: Image display error - content type: {type(content)}")
        print(f"DEBUG: Image display error - content length: {len(str(content)) if hasattr(content, '__len__') else 'No length'}")


def _display_json_file_inline(content, file_name, unique_key=None):
    """Display JSON content inline."""
    try:
        import json
        import time
        
        # Convert content to string if it's bytes
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Parse and pretty-print JSON
        json_data = json.loads(content)
        st.json(json_data)
        
    except Exception as e:
        st.error(f"Error parsing JSON {file_name}: {e}")
        # Fallback to code display
        st.code(content[:1000] + "..." if len(content) > 1000 else content)


def _display_text_file_inline(content, file_name, unique_key=None):
    """Display text content inline."""
    try:
        import time
        # Convert content to string if it's bytes
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Display as code block
        st.code(content[:2000] + "..." if len(content) > 2000 else content)
        
    except Exception as e:
        st.error(f"Error displaying text {file_name}: {e}")


def _is_likely_file_content(result_str, tool_name):
    """Check if the tool result is likely file content."""
    # Check for common file patterns
    file_indicators = [
        # CSV indicators
        "," in result_str and "\n" in result_str and len(result_str.split("\n")) > 2,
        # Image indicators (base64)
        result_str.startswith("data:image/") or (len(result_str) > 100 and result_str.isalnum()),
        # JSON indicators
        result_str.strip().startswith("{") and result_str.strip().endswith("}"),
        result_str.strip().startswith("[") and result_str.strip().endswith("]"),
        # Code indicators
        "def " in result_str or "import " in result_str or "class " in result_str,
        # File path indicators
        "/" in result_str and ("." in result_str) and any(ext in result_str.lower() for ext in [".csv", ".png", ".jpg", ".json", ".txt"]),
    ]
    
    return any(file_indicators)


def _extract_file_info_from_result(result_str, tool_name):
    """Extract file information from tool result."""
    import re
    import base64
    
    # Try to detect file type and content
    file_info = {
        "name": f"generated_{tool_name}_{len(st.session_state.get('generated_files', []))}",
        "content": result_str,
        "type": "unknown"
    }
    
    # Detect CSV
    if "," in result_str and "\n" in result_str:
        lines = result_str.split("\n")
        if len(lines) > 2 and all("," in line for line in lines[:3]):
            file_info["name"] = f"data_{len(st.session_state.get('generated_files', []))}.csv"
            file_info["type"] = "csv"
            return file_info
    
    # Detect JSON
    if result_str.strip().startswith(("{", "[")):
        try:
            import json
            json.loads(result_str)
            file_info["name"] = f"data_{len(st.session_state.get('generated_files', []))}.json"
            file_info["type"] = "json"
            return file_info
        except:
            pass
    
    # Detect image (base64)
    if result_str.startswith("data:image/"):
        try:
            header, base64_content = result_str.split(",", 1)
            mime_type = header.split(";")[0].replace("data:", "")
            ext = mime_type.split("/")[-1]
            file_info["name"] = f"image_{len(st.session_state.get('generated_files', []))}.{ext}"
            file_info["type"] = mime_type
            file_info["content"] = base64_content
            return file_info
        except:
            pass
    
    # Detect Python code
    if any(keyword in result_str for keyword in ["def ", "import ", "class ", "if __name__"]):
        file_info["name"] = f"code_{len(st.session_state.get('generated_files', []))}.py"
        file_info["type"] = "python"
        return file_info
    
    # Detect text file
    if len(result_str) > 50 and not result_str.startswith(("{", "[", "data:")):
        file_info["name"] = f"text_{len(st.session_state.get('generated_files', []))}.txt"
        file_info["type"] = "text"
        return file_info
    
    return None


def _display_files_in_answer(files):
    """Display files directly in the answer section based on file type."""
    if not files:
        return
    
    st.markdown("---")
    st.markdown("### 📁 Generated Files")
    
    for i, file_info in enumerate(files):
        file_name = file_info.get("name", f"file_{i}")
        file_content = file_info.get("content", "")
        file_type = file_info.get("type", "unknown")
        
        st.markdown(f"**{file_name}** ({file_type})")
        
        try:
            if file_type.lower() in ["csv", "text/csv"]:
                _display_csv_file(file_content, file_name)
            elif file_type.lower() in ["png", "jpg", "jpeg", "gif", "bmp", "image/png", "image/jpeg", "image/gif"]:
                _display_image_file(file_content, file_name)
            elif file_type.lower() in ["json", "application/json"]:
                _display_json_file(file_content, file_name)
            elif file_type.lower() in ["txt", "text/plain"]:
                _display_text_file(file_content, file_name)
            else:
                # Default: show as downloadable file
                st.download_button(
                    label=f"📥 Download {file_name}",
                    data=file_content if isinstance(file_content, bytes) else file_content.encode(),
                    file_name=file_name,
                    mime="application/octet-stream",
                    key=f"download_legacy_{file_name}_{i}"
                )
        except Exception as e:
            st.error(f"Error displaying file {file_name}: {e}")
            # Fallback to download button
            st.download_button(
                label=f"📥 Download {file_name}",
                data=file_content if isinstance(file_content, bytes) else file_content.encode(),
                file_name=file_name,
                mime="application/octet-stream",
                key=f"download_error_{file_name}_{i}"
            )


def _display_csv_file(content, file_name):
    """Display CSV content as a table."""
    try:
        import pandas as pd
        import io
        
        # Convert content to string if it's bytes
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Read CSV from string
        df = pd.read_csv(io.StringIO(content))
        
        # Display the dataframe
        st.dataframe(df, use_container_width=True)
        
        # Also provide download button
        import time
        unique_key = unique_key or f"{file_name}_{int(time.time() * 1000)}"
        st.download_button(
            label=f"📥 Download {file_name}",
            data=content.encode(),
            file_name=file_name,
            mime="text/csv",
            key=f"download_csv_{unique_key}"
        )
    except Exception as e:
        st.error(f"Error parsing CSV {file_name}: {e}")
        # Fallback to text display
        st.code(content[:1000] + "..." if len(content) > 1000 else content)


def _display_image_file(content, file_name):
    """Display image content using data URI."""
    try:
        import base64
        
        print(f"DEBUG: _display_image_file - content type: {type(content)}")
        print(f"DEBUG: _display_image_file - content length: {len(str(content)) if hasattr(content, '__len__') else 'No length'}")
        
        # Get base64 string
        if isinstance(content, str):
            if content.startswith("data:"):
                # Already a data URI
                data_uri = content
            else:
                # Plain base64 string, create data URI
                data_uri = f"data:image/png;base64,{content}"
        elif hasattr(content, 'getvalue'):
            # BytesIO object, convert to base64
            import io
            content_bytes = content.getvalue()
            base64_str = base64.b64encode(content_bytes).decode('utf-8')
            data_uri = f"data:image/png;base64,{base64_str}"
        elif isinstance(content, io.BytesIO):
            # BytesIO object, convert to base64
            content_bytes = content.getvalue()
            base64_str = base64.b64encode(content_bytes).decode('utf-8')
            data_uri = f"data:image/png;base64,{base64_str}"
        elif isinstance(content, bytes):
            # Raw bytes, convert to base64
            base64_str = base64.b64encode(content).decode('utf-8')
            data_uri = f"data:image/png;base64,{base64_str}"
        else:
            # Try to convert to string then to base64
            content_str = str(content)
            base64_str = base64.b64encode(content_str.encode('utf-8')).decode('utf-8')
            data_uri = f"data:image/png;base64,{base64_str}"
        
        print(f"DEBUG: _display_image_file - data URI length: {len(data_uri)}")
        
        # Display the image using markdown with data URI
        st.markdown(f"![{file_name}]({data_uri})", unsafe_allow_html=True)
        
        # Also provide download button
        import time
        unique_key = unique_key or f"{file_name}_{int(time.time() * 1000)}"
        st.download_button(
            label=f"📥 Download {file_name}",
            data=content,
            file_name=file_name,
            mime="image/png",
            key=f"download_image_{unique_key}"
        )
    except Exception as e:
        st.error(f"Error displaying image {file_name}: {e}")


def _display_json_file(content, file_name):
    """Display JSON content in a formatted way."""
    try:
        import json
        
        # Convert content to string if it's bytes
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Parse and pretty-print JSON
        json_data = json.loads(content)
        st.json(json_data)
        
        # Also provide download button
        import time
        unique_key = unique_key or f"{file_name}_{int(time.time() * 1000)}"
        st.download_button(
            label=f"📥 Download {file_name}",
            data=content.encode(),
            file_name=file_name,
            mime="application/json",
            key=f"download_json_{unique_key}"
        )
    except Exception as e:
        st.error(f"Error parsing JSON {file_name}: {e}")
        # Fallback to code display
        st.code(content[:1000] + "..." if len(content) > 1000 else content)


def _display_text_file(content, file_name):
    """Display text content."""
    try:
        # Convert content to string if it's bytes
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Display as code block
        st.code(content[:2000] + "..." if len(content) > 2000 else content)
        
        # Also provide download button
        import time
        unique_key = unique_key or f"{file_name}_{int(time.time() * 1000)}"
        st.download_button(
            label=f"📥 Download {file_name}",
            data=content.encode(),
            file_name=file_name,
            mime="text/plain",
            key=f"download_text_{unique_key}"
        )
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

    # Create streaming config
    streaming_config = StreamingConfig(enabled=streaming_enabled, mode=StreamingMode.ALL)

    agent = ReActAgent(
        name="DataAnalyst",
        id="DataAnalyst",
        llm=llm,
        tools=[tool],
        role=AGENT_ROLE,
        max_loops=20,
        inference_mode=InferenceMode.XML,
        streaming=streaming_config,
    )

    return agent


def analyze_csv_file(file_obj: io.BytesIO, analysis_prompt: str = None, streaming_enabled: bool = False) -> tuple[str, dict]:
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
        # Initialize session state for loading
        if "loading_placeholder" not in st.session_state:
            st.session_state.loading_placeholder = None
        if "is_loading" not in st.session_state:
            st.session_state.is_loading = False
        
        # Show loading spinner when analysis starts
        if streaming_enabled:
            _show_loading_spinner()
            st.session_state.is_loading = True
        
        agent = create_agent(streaming_enabled=streaming_enabled)
        
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
            
            IMPORTANT: Always save your visualizations as PNG files and reference them in your response using [filename.png] format so they display inline.
            """
        
        analysis_prompt = "return in the ouput just ![sample_data.csv]"
        result = agent.run(
            input_data={"input": analysis_prompt, "files": [file_obj]},
        )

        content = result.output.get("content", "")
        files = result.output.get("files", {})
        
        return content, files
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return f"Error during analysis: {str(e)}", {}


def run_agent_with_streaming(file_obj: io.BytesIO, analysis_prompt: str, send_handler: AsyncStreamingIteratorCallbackHandler) -> tuple[str, dict]:
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
            
            IMPORTANT: Always save your visualizations as PNG files and reference them in your response using [filename.png] format so they display inline.
            """
        
        flow = Workflow(
            flow=Flow(nodes=[agent]),
        )
        
        result = flow.run(
            input_data={"input": analysis_prompt, "files": [file_obj]},
            config=RunnableConfig(callbacks=[send_handler])
        )

        content = result.output[agent.id]["output"].get("content", "")
        files = result.output[agent.id]["output"].get("files", {})
        
        
        return content, files
        
    except Exception as e:
        logger.error(f"Streaming analysis failed: {e}")
        return f"Error during analysis: {str(e)}", {}


def run_agent_with_streaming_sync(file_obj: io.BytesIO, analysis_prompt: str) -> tuple[str, dict]:
    """
    
    The use_column_width parameter has been deprecated and will be removed in a future release. Please utilize the use_container_width parameter instead

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


def run_agent_sync(file_obj: io.BytesIO, analysis_prompt: str, send_handler: AsyncStreamingIteratorCallbackHandler) -> tuple[str, dict]:
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
            
            IMPORTANT: Always save your visualizations as PNG files and reference them in your response using [filename.png] format so they display inline.
            """
        
        flow = Workflow(
            flow=Flow(nodes=[agent]),
        )

        result = flow.run(
            input_data={"input": analysis_prompt, "files": [file_obj]}, 
            config=RunnableConfig(callbacks=[send_handler])
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

        return response, files
        
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
        print(f"delta {delta}")
        
        # Handle different delta formats
        if isinstance(delta, str):
            # Delta is just a string (final answer)
            print("DEBUG: Delta is string, treating as final answer")
            # Only call callback if this is new content
            if "last_answer_content" not in st.session_state or st.session_state.last_answer_content != delta:
                st.session_state.last_answer_content = delta
                streamlit_callback("answer", delta)
        elif isinstance(delta, dict):
            # Delta is a dictionary with step and content
            step = delta.get("step", "")
            print(f"step {step}")
            content = delta.get("content", "")
            
            if step and content:
                streamlit_callback(step, content)
            elif content:
                # Content without step, treat as answer
                print("DEBUG: Content without step, treating as answer")
                # Only call callback if this is new content
                if "last_answer_content" not in st.session_state or st.session_state.last_answer_content != content:
                    st.session_state.last_answer_content = content
                    streamlit_callback("answer", content)
        else:
            print(f"DEBUG: Unknown delta type: {type(delta)}")


def save_analysis_files(files: dict | list, output_dir: str = "./analysis_outputs") -> List[str]:
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
                file_description = getattr(file_bytesio, "description", "Generated file")
                
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

    return saved_files
