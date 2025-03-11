import html
import time

import streamlit as st
from backend import generate_agent_response, setup_agent

st.sidebar.title("Agent Configuration")
streaming_enabled = st.sidebar.checkbox("Enable Streaming", value=False)

streaming_mode = st.sidebar.radio("Streaming Mode", options=["Steps", "Answer"], index=0)
if "agent" not in st.session_state or st.sidebar.button("Apply Changes"):
    st.session_state.agent = setup_agent(streaming_enabled, streaming_mode)
    st.session_state.messages = []
    st.session_state.chunk_color_map = {}
    st.session_state.color_palette = [
        "#228be6",
        "#fa5252",
        "#40c057",
        "#fcc419",
        "#be4bdb",
        "#15aabf",
        "#e64980",
        "#82c91e",
        "#fab005",
        "#4c6ef5",
        "#7950f2",
        "#e67700",
    ]
    st.session_state.color_index = 0

st.title("Orchestrator Chat")
st.write("Ask questions and get responses from an AI assistant.")


# Function to get a color for a chunk type
def get_color_for_type(chunk_type):
    if chunk_type not in st.session_state.chunk_color_map:
        color = st.session_state.color_palette[st.session_state.color_index % len(st.session_state.color_palette)]
        st.session_state.chunk_color_map[chunk_type] = color
        st.session_state.color_index += 1
    return st.session_state.chunk_color_map[chunk_type]


def get_colored_chunk(content, chunk_type):
    safe_content = html.escape(content)
    color = get_color_for_type(chunk_type)
    return f'<span style="color: {color};">{safe_content}</span>'


# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Input and response processing
if user_input := st.chat_input("You: "):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        st.session_state.messages.append({"role": "assistant", "content": ""})

        for chunk in generate_agent_response(st.session_state.agent, user_input):
            chunk_content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            chunk_type = chunk.get("choices", [{}])[0].get("delta", {}).get("type", "default")

            colored_chunk = get_colored_chunk(chunk_content, chunk_type)

            full_response += " " + colored_chunk
            message_placeholder.markdown(full_response, unsafe_allow_html=True)

            st.session_state.messages[-1]["content"] = full_response

            time.sleep(0.05)

        message_placeholder.markdown(full_response, unsafe_allow_html=True)

        st.session_state.messages[-1]["content"] = full_response

    st.session_state["new_input"] = ""
