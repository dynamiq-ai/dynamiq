import streamlit as st
from runner import create_orchestrator, run_orchestrator
import yaml

def process_file(uploaded_file):
    content = uploaded_file.read().decode("utf-8")
    return content

st.title("Kiroku Paper/Report Writer")

st.write("This app allows you to generate paper/report based on configuration and feedback.")

uploaded_file = st.file_uploader("Upload config file (yaml)", type=["yaml"], key="file_uploader")

if uploaded_file:
    with st.spinner("Generating..."):
        file_contents = uploaded_file.getvalue().decode("utf-8")
        configuration_context = yaml.safe_load(file_contents)
        orchestrator = create_orchestrator(configuration_context)
        result = run_orchestrator(orchestrator, configuration_context)
        
    st.text_input("Type in feedback")
    file_content = process_file(uploaded_file)
    st.text_area("Generated result:", result, height=200)