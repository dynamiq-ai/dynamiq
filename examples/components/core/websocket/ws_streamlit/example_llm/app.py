import json
import queue
import threading

import streamlit as st
import websocket

from dynamiq.utils.logger import logger
from examples.components.core.websocket.ws_streamlit.example_llm.server import (
    HOST,
    OPENAI_NODE_STREAMING_EVENT,
    PORT,
    WF_ID,
)


@st.cache_resource
def get_queue():
    return queue.Queue()


message_queue = get_queue()


# WebSocket callbacks
def on_message(ws, message):
    data = json.loads(message)
    logger.info(f"Received message: {json.dumps(data, indent=2)}")
    if data["event"] == OPENAI_NODE_STREAMING_EVENT:
        content = data.get("data", {}).get("choices", [{}])[0].get("delta", {}).get("content")
        if content:
            message_queue.put(content)
    elif data["event"] == "end":
        message_queue.put("DONE")


def on_error(ws, error):
    logger.error(f"WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")


@st.cache_resource
def create_ws_connection():
    ws_app_conn = websocket.WebSocketApp(
        f"ws://{HOST}:{PORT}/workflows/test", on_message=on_message, on_error=on_error, on_close=on_close
    )

    # Run WebSocket connection in a separate thread
    wst = threading.Thread(target=ws_app_conn.run_forever)
    wst.daemon = True
    wst.start()

    # Wait for WebSocket connection to be established
    while not ws_app_conn.sock or not ws_app_conn.sock.connected:
        pass

    logger.info("WebSocket connection established")
    return ws_app_conn


ws_conn = create_ws_connection()


# Streamlit app
def main():
    st.title("Chat with AI Assistant")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What is your question?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add assistant message placeholder
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Prepare message for server
            workflow_input = {
                "event": "start",
                "entity_id": WF_ID,
                "data": {"input": prompt},
            }

            # Log the input being sent to the workflow
            logger.info(f"Sending input to workflow: {json.dumps(workflow_input, indent=2)}")

            # Send message to server
            ws_conn.send(json.dumps(workflow_input))

            # Process messages from the queue
            while True:
                try:
                    msg = message_queue.get(timeout=0.1)
                    if msg == "DONE":
                        break
                    full_response += msg
                    message_placeholder.markdown(full_response + "▌")
                except queue.Empty:
                    continue

            # Display final response
            message_placeholder.markdown(full_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
