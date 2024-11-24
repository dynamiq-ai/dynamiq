import streamlit as st
import yaml
from runner import create_orchestrator, run_orchestrator


def process_file(uploaded_file):
    content = uploaded_file.read().decode("utf-8")
    return content


st.title("Paper/Report/Document Writer")

st.write("This app allows you to generate paper/report based on configuration and feedback.")

uploaded_file = st.file_uploader("Upload config file (yaml)", type=["yaml"], key="file_uploader")

if uploaded_file:
    if "result" not in st.session_state:
        with st.spinner("Generating..."):
            file_contents = uploaded_file.getvalue().decode("utf-8")
            configuration_context = yaml.safe_load(file_contents)
            orchestrator = create_orchestrator(configuration_context)
            run_orchestrator(orchestrator)
            st.session_state.result = orchestrator.context.get("draft").replace("```markdown", "").replace("```", "")
            st.session_state.orchestrator = orchestrator

    feedback = st.text_input("Type in feedback")

    if st.button("Update"):
        with st.spinner("Updating..."):
            orchestrator = st.session_state.orchestrator
            orchestrator.context["update_instruction"] = feedback
            orchestrator = st.session_state.orchestrator
            run_orchestrator(orchestrator)
            st.session_state.result = orchestrator.context.get("draft").replace("```markdown", "").replace("```", "")
            st.session_state.orchestrator = orchestrator

    st.write(st.session_state.result)
