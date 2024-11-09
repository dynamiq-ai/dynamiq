import time

import streamlit as st

from examples.use_case_search.server import process_query


def reset_conversation():
    """
    Resets the conversation state, clearing query history and input fields.
    """
    st.session_state.query_history = []
    st.session_state.current_query = ""
    st.session_state.clear_input = False


def initialize_session_state():
    """
    Initializes session state variables if not already set.
    """
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False


def handle_query_submission(query: str):
    """
    Handles query submission and displays the result chunk by chunk.

    Args:
        query (str): The user input query.
    """
    if query.strip():
        st.session_state.query_history.append(query)
        st.session_state.current_query = query

        result_placeholder = st.empty()

        with st.spinner("Processing your query..."):
            result_text = ""
            for chunk in process_query(query):
                result_text += chunk + " "
                result_placeholder.write(result_text.strip())
                time.sleep(0.05)

        st.session_state.clear_input = True
    else:
        st.warning("Please enter a query before submitting.")


def main():
    st.title("Search Application")

    initialize_session_state()

    query = st.text_input("Enter your query:", value="", key="initial_input")

    if st.button("Submit"):
        handle_query_submission(query)

    if st.button("Start New Query"):
        reset_conversation()
        st.experimental_rerun()

    if st.session_state.clear_input:
        st.session_state.clear_input = False


if __name__ == "__main__":
    main()
