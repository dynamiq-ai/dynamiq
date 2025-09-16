import io
import os
import time
import streamlit as st
from agent import analyze_csv_file, save_analysis_files, run_agent_with_streaming_sync

st.set_page_config(
    page_title="Data Analyst Chat Agent",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "generated_files" not in st.session_state:
    st.session_state.generated_files = []
if "streaming_enabled" not in st.session_state:
    st.session_state.streaming_enabled = True
if "step_placeholders" not in st.session_state:
    st.session_state.step_placeholders = {}
if "step_contents" not in st.session_state:
    st.session_state.step_contents = {}
if "step_order" not in st.session_state:
    st.session_state.step_order = []
if "current_loop_nums" not in st.session_state:
    st.session_state.current_loop_nums = {}


def init_stream_ui(reset: bool = False):
    """Initialize or reset dynamic step boxes for streaming UI (matching agents_test.py)."""
    if reset or "step_placeholders" not in st.session_state:
        st.session_state.step_placeholders = {}
        st.session_state.step_contents = {}
        st.session_state.step_order = []
        st.session_state.current_config = None
        st.session_state.current_loop_nums = {}


def clear_stream_ui():
    """Clear all streaming UI elements (matching agents_test.py)."""
    st.session_state.step_placeholders = {}
    st.session_state.step_contents = {}
    st.session_state.step_order = []
    st.session_state.current_loop_nums = {}

st.title("💬 Data Analyst Chat Agent")
st.markdown("Upload a CSV or text file and chat with an AI data analyst!")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload File")
    uploaded_file = st.file_uploader(
        "Choose a CSV or text file",
        type=["csv", "txt"],
        help="Upload a CSV or text file for analysis",
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        # Convert uploaded file to BytesIO and store in session state
        file_obj = io.BytesIO(uploaded_file.getvalue())
        file_obj.name = uploaded_file.name
        file_obj.description = f"File: {uploaded_file.name}"
        st.session_state.uploaded_file = file_obj
        
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.info(f"File size: {uploaded_file.size:,} bytes")
        
        # Clear previous messages when new file is uploaded
        if "previous_file" not in st.session_state or st.session_state.previous_file != uploaded_file.name:
            st.session_state.messages = []
            st.session_state.generated_files = []
            st.session_state.previous_file = uploaded_file.name
            st.rerun()
    
    st.header("⚙️ Settings")
    streaming_enabled = st.checkbox(
        "🔄 Enable Streaming", 
        value=st.session_state.streaming_enabled,
        help="Show real-time agent reasoning and responses"
    )
    st.session_state.streaming_enabled = streaming_enabled
    
    st.header("💡 Example Prompts")
    example_prompts = [
        "Give me a basic overview of this data",
        "Create visualizations for the most important columns",
        "Find correlations between numeric columns",
        "Identify outliers in the data",
        "Compare different groups/categories",
        "Generate summary statistics",
        "Create a correlation matrix",
        "Show me the distribution of key variables"
    ]
    
    for prompt in example_prompts:
        if st.button(f"💬 {prompt}", key=f"example_{prompt}"):
            st.session_state.user_input = prompt
            st.rerun()

# Main chat interface
if st.session_state.uploaded_file is not None:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show generated files for assistant messages
            if message["role"] == "assistant" and message.get("files"):
                st.markdown("**📁 Generated Files:**")
                
                # Handle list of BytesIO objects (new format)
                if isinstance(message["files"], list):
                    for file_bytesio in message["files"]:
                        try:
                            # Get file metadata from BytesIO object
                            file_name = getattr(file_bytesio, "name", f"file_{id(file_bytesio)}.bin")
                            
                            # Read content from BytesIO
                            file_data = file_bytesio.read()
                            file_bytesio.seek(0)  # Reset position for potential future reads
                            
                            st.download_button(
                                label=f"📥 Download {file_name}",
                                data=file_data,
                                file_name=file_name,
                                mime="application/octet-stream",
                                key=f"download_{file_name}_{len(st.session_state.messages)}"
                            )
                        except Exception as e:
                            st.error(f"Error preparing file {getattr(file_bytesio, 'name', 'unknown')}: {e}")
                
                # Handle dictionary format (legacy)
                elif isinstance(message["files"], dict):
                    for file_path, file_content in message["files"].items():
                        try:
                            if isinstance(file_content, str) and file_content.startswith("data:"):
                                import base64
                                header, base64_content = file_content.split(",", 1)
                                file_data = base64.b64decode(base64_content)
                                mime_type = header.split(";")[0].replace("data:", "")
                            else:
                                # Handle base64 content - convert to data URI for proper display
                                import base64
                                file_data = base64.b64decode(file_content) if isinstance(file_content, str) else file_content
                                
                                # Detect MIME type from file extension
                                file_ext = os.path.splitext(file_path)[1].lower()
                                mime_map = {
                                    '.png': 'image/png',
                                    '.jpg': 'image/jpeg', 
                                    '.jpeg': 'image/jpeg',
                                    '.gif': 'image/gif',
                                    '.svg': 'image/svg+xml',
                                    '.pdf': 'application/pdf',
                                    '.csv': 'text/csv',
                                    '.txt': 'text/plain',
                                    '.json': 'application/json'
                                }
                                mime_type = mime_map.get(file_ext, 'application/octet-stream')
                            
                            # For images, try to display them inline
                            if mime_type.startswith('image/'):
                                try:
                                    st.image(file_data, caption=os.path.basename(file_path), use_column_width=True)
                                except Exception as img_error:
                                    st.warning(f"Could not display image {os.path.basename(file_path)}: {img_error}")
                            
                            # Always provide download button
                            st.download_button(
                                label=f"📥 Download {os.path.basename(file_path)}",
                                data=file_data,
                                file_name=os.path.basename(file_path),
                                mime=mime_type,
                                key=f"download_{file_path}_{len(st.session_state.messages)}"
                            )
                        except Exception as e:
                            st.error(f"Error preparing file {file_path}: {e}")
    
    # Chat input
    if prompt := st.chat_input("Ask the data analyst anything about your data..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            if st.session_state.streaming_enabled:
                # Initialize streaming UI without clearing previous content
                init_stream_ui(reset=False)
                
                # Run streaming analysis with REAL intermediate steps
                try:
                    # Use the proper streaming function with real agent reasoning
                    analysis_result, generated_files = run_agent_with_streaming_sync(
                        st.session_state.uploaded_file, prompt
                    )
                    
                except Exception as e:
                    st.error(f"Streaming failed: {e}")
                    import traceback
                    st.error(f"Full error: {traceback.format_exc()}")
                    st.warning("🔄 Falling back to non-streaming mode...")
                    # Fallback to non-streaming
                    analysis_result, generated_files = analyze_csv_file(st.session_state.uploaded_file, prompt, streaming_enabled=False)
            else:
                # Non-streaming analysis
                with st.spinner("🤖 AI Agent is analyzing your data..."):
                    analysis_result, generated_files = analyze_csv_file(st.session_state.uploaded_file, prompt, streaming_enabled=False)
            
            # Save generated files
            if generated_files:
                saved_files = save_analysis_files(generated_files)
                if saved_files:
                    st.success(f"📁 Generated {len(saved_files)} analysis files!")
                    for file_path in saved_files:
                        st.info(f"💾 Saved: {file_path}")
            
            # Display final results only if not streaming (streaming UI handles this)
            if not st.session_state.streaming_enabled:
                st.markdown("## 📈 Final Analysis")
                st.markdown(analysis_result)
            
            # Show generated files for download only if not streaming (streaming UI handles this)
            if generated_files and not st.session_state.streaming_enabled:
                st.markdown("**📁 Generated Files:**")
                
                # Handle list of BytesIO objects (new format)
                if isinstance(generated_files, list):
                    for file_bytesio in generated_files:
                        try:
                            # Get file metadata from BytesIO object
                            file_name = getattr(file_bytesio, "name", f"file_{id(file_bytesio)}.bin")
                            
                            # Read content from BytesIO
                            file_data = file_bytesio.read()
                            file_bytesio.seek(0)  # Reset position for potential future reads
                            
                            st.download_button(
                                label=f"📥 Download {file_name}",
                                data=file_data,
                                file_name=file_name,
                                mime="application/octet-stream",
                                key=f"download_{file_name}_{len(st.session_state.messages)}"
                            )
                        except Exception as e:
                            st.error(f"Error preparing file {getattr(file_bytesio, 'name', 'unknown')}: {e}")
                
                # Handle dictionary format (legacy)
                elif isinstance(generated_files, dict):
                    for file_path, file_content in generated_files.items():
                        try:
                            if isinstance(file_content, str) and file_content.startswith("data:"):
                                import base64
                                header, base64_content = file_content.split(",", 1)
                                file_data = base64.b64decode(base64_content)
                                mime_type = header.split(";")[0].replace("data:", "")
                            else:
                                # Handle base64 content - convert to data URI for proper display
                                import base64
                                file_data = base64.b64decode(file_content) if isinstance(file_content, str) else file_content
                                
                                # Detect MIME type from file extension
                                file_ext = os.path.splitext(file_path)[1].lower()
                                mime_map = {
                                    '.png': 'image/png',
                                    '.jpg': 'image/jpeg', 
                                    '.jpeg': 'image/jpeg',
                                    '.gif': 'image/gif',
                                    '.svg': 'image/svg+xml',
                                    '.pdf': 'application/pdf',
                                    '.csv': 'text/csv',
                                    '.txt': 'text/plain',
                                    '.json': 'application/json'
                                }
                                mime_type = mime_map.get(file_ext, 'application/octet-stream')
                            
                            # For images, try to display them inline
                            if mime_type.startswith('image/'):
                                try:
                                    st.image(file_data, caption=os.path.basename(file_path), use_column_width=True)
                                except Exception as img_error:
                                    st.warning(f"Could not display image {os.path.basename(file_path)}: {img_error}")
                            
                            # Always provide download button
                            st.download_button(
                                label=f"📥 Download {os.path.basename(file_path)}",
                                data=file_data,
                                file_name=os.path.basename(file_path),
                                mime=mime_type,
                                key=f"download_{file_path}_{len(st.session_state.messages)}"
                            )
                        except Exception as e:
                            st.error(f"Error preparing file {file_path}: {e}")
            
            # End of file display section for non-streaming mode
        
        # Add assistant message to chat
        # st.session_state.messages.append({
        #     "role": "assistant", 
        #     "content": analysis_result,
        #     "files": generated_files
        # })
        
        # Update generated files in session state
        print(f"DEBUG: Generated files type: {type(generated_files)}")
        print(f"DEBUG: Generated files content: {generated_files}")
        
        if isinstance(generated_files, dict):
            # Convert dict to list format
            for file_name, file_content in generated_files.items():
                file_info = {
                    "name": file_name,
                    "content": file_content,
                    "type": "unknown"
                }
                st.session_state.generated_files.append(file_info)
        elif isinstance(generated_files, list):
            # Add list of files to generated_files
            for file_bytesio in generated_files:
                file_info = {
                    "name": getattr(file_bytesio, "name", f"generated_file_{len(st.session_state.generated_files)}.bin"),
                    "content": file_bytesio.read(),
                    "type": "unknown"
                }
                file_bytesio.seek(0)  # Reset position
                st.session_state.generated_files.append(file_info)
        else:
            print(f"DEBUG: Unexpected files type: {type(generated_files)}")
            st.warning(f"Unexpected files format: {type(generated_files)}")
        
        # Rerun to update the chat
        st.rerun()

else:
    # Welcome message
    st.markdown("""
    ## Welcome to the Data Analyst Chat Agent! 🚀
    
    This AI-powered chat tool can help you:
    - 📊 Analyze CSV data with statistical summaries
    - 📈 Create visualizations and charts
    - 🔍 Identify patterns and insights
    - 💡 Provide data-driven recommendations
    - 💬 Have interactive conversations about your data
    
    ### How to use:
    1. Upload a CSV file using the sidebar
    2. Start chatting with the AI data analyst
    3. Ask questions, request specific analyses, or ask for visualizations
    4. Download generated files and continue the conversation
    
    ### Example Questions:
    - "What are the main insights from this data?"
    - "Create a scatter plot of salary vs experience"
    - "Which department has the highest average salary?"
    - "Are there any outliers I should be aware of?"
    - "Show me the correlation between age and performance"
    """)
    
    # Sample data info
    st.info("💡 Tip: Upload a CSV file to start chatting with the AI data analyst!")