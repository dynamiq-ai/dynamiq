import time

import streamlit as st

from examples.use_case_search.server_via_dynamiq import process_query


def reset_conversation():
    st.session_state.query_history = []
    st.session_state.current_query = ""
    st.session_state.clear_input = False


def main():
    st.title("Search")

    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False

    query = st.text_input("Enter your query:", value="" if st.session_state.clear_input else "", key="initial_input")

    # Handle query submission
    if st.button("Submit"):
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

    if st.button("Start New Query"):
        reset_conversation()
        st.experimental_rerun()

    if st.session_state.clear_input:
        st.session_state.clear_input = False


# Run the Streamlit app
if __name__ == "__main__":
    main()
