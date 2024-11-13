import time

import streamlit as st
from backend import generate_agent_response, setup_agent

st.sidebar.title("Agent Configuration")
agent_role = st.sidebar.text_input("Agent Role", "helpful assistant")
streaming_enabled = st.sidebar.checkbox("Enable Streaming", value=False)

if "agent" not in st.session_state or st.sidebar.button("Apply Changes"):
    st.session_state.agent = setup_agent(agent_role, streaming_enabled)
    st.session_state.messages = []

st.title("Simple Agent Chat")
st.write("Ask questions and get responses from an AI assistant.")

# Display chat history from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("You: "):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        st.session_state.messages.append({"role": "assistant", "content": ""})

        for chunk in generate_agent_response(st.session_state.agent, user_input):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

            st.session_state.messages[-1]["content"] = full_response

            time.sleep(0.05)
        message_placeholder.markdown(full_response)

        st.session_state.messages[-1]["content"] = full_response

    st.session_state["new_input"] = ""
