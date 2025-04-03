import asyncio

import streamlit as st

from examples.components.agents.streaming.intermediate_streaming.graph_orchestrator.graph_orchestrator import (
    run_orchestrator_async,
)

if __name__ == "__main__":
    st.markdown("# Email Write Orchestrator")

    with st.form("my_form"):
        request = st.text_input("What is your request", placeholder="Write email about party invitation.")

        submitted = st.form_submit_button("Submit")

    if submitted:
        with st.status("🤖 **Agents at work...**", state="running", expanded=True) as status:
            with st.container(height=250, border=False):
                result = asyncio.run(run_orchestrator_async(request))
            status.update(label="✅ Result is ready!", state="complete", expanded=False)

        st.subheader("Generated result", anchor=False, divider="rainbow")
        st.markdown(result)
