import time

import streamlit as st
from backend import agent, generate_agent_response, read_file_as_bytesio

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "new_input" not in st.session_state:
    st.session_state.new_input = ""

# Sidebar Configuration
st.sidebar.title("Files")

st.sidebar.markdown("---")
uploaded_files = st.sidebar.file_uploader("Upload files (optional)", type=None, accept_multiple_files=False)

# Main Chat Interface
st.title("ReAct Agent Chat")
st.write("Ask questions, upload files, and interact with an intelligent assistant.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Handling
if user_input := st.chat_input("You: "):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Add placeholder for assistant message
        st.session_state.messages.append({"role": "assistant", "content": ""})

        # Process uploaded file(s)
        files_to_process = [read_file_as_bytesio(uploaded_files)] if uploaded_files else None

        # Generate and stream response
        for chunk in generate_agent_response(agent, user_input, files=files_to_process):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
            st.session_state.messages[-1]["content"] = full_response
            time.sleep(0.05)

        message_placeholder.markdown(full_response)
        st.session_state.messages[-1]["content"] = full_response

    st.session_state.new_input = ""

if uploaded_files:
    if isinstance(uploaded_files, list):  # Multiple files
        st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s) successfully!")
    else:  # Single file
        st.sidebar.success(f"Uploaded file: {uploaded_files.name} successfully!")
else:
    st.sidebar.info("No files uploaded. You can upload a file to enhance the assistant's capabilities.")
