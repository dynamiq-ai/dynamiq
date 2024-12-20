import time

import streamlit as st
from backend import generate_agent_response, setup_agent

if "agent" not in st.session_state or st.sidebar.button("Reset Agent"):
    st.session_state.agent = setup_agent()
    st.session_state.messages = []

st.sidebar.title("Dynamiq Assistant")
st.sidebar.info(
    """
    🤖 **Your ERP System Assistant**
    Simplify database interactions, retrieve insightful summaries, and keep your ERP data organized.
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("📞 Support: +1-800-ERP-HELP")
st.sidebar.markdown("🌐 [Visit Us](https://example.com)")

# Title and Header
st.title("Dynamiq Assistant Chat")
st.subheader("Your Smart ERP System Assistant 🤝")
st.write(
    """
    Ask me anything about your ERP system.
    I can help you retrieve data,
    validate queries, and guide you through managing your inventory, orders, and more!
    """
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_input := st.chat_input("Ask your ERP Assistant..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Append a temporary empty message for the assistant
        st.session_state.messages.append({"role": "assistant", "content": ""})

        for chunk in generate_agent_response(st.session_state.agent, user_input):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

            st.session_state.messages[-1]["content"] = full_response

            time.sleep(0.05)
        message_placeholder.markdown(full_response)

        st.session_state.messages[-1]["content"] = full_response

st.markdown("---")
st.markdown(
    """
    **🤖 Dynamiq Assistant**
    Designed to streamline your ERP management.
    🌟Powered by Dynamiq Framework.
    """
)
