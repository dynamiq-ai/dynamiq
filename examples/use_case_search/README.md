## Search use - case

This project demonstrates two different approaches to building a web-based search application powered by **Dynamiq**. The main goal is to showcase how Dynamiq's workflows, agents, and tools can be used in a flexible manner to process user queries and provide synthesized, accurate search results in real-time.

The project has two stages:
1. **Stage 1**: Building the logic server programmatically using Dynamiq code, where we define nodes, agents, and tools, and then craft a workflow. The workflow is then integrated into a frontend app using Streamlit to create an interactive web search tool.
2. **Stage 2**: Using the **Dynamiq UI** to craft nodes, agents, and tools visually and deploy the logic via an API endpoint. This endpoint is then used as the backend for a Streamlit-based web application.

---

## Project Structure

```
README.md
app.py
run.sh
server.py
server_via_dynamiq.py
```

### Files Overview

- **README.md**: This file.
- **app.py**: The main entry point for the Streamlit app. It handles query input, processes queries using the server logic, and displays results in real-time.
- **run.sh**: A script to launch the application (optional, depending on setup).
- **server.py**: Contains the server logic for **Stage 1**, where Dynamiq workflows are defined programmatically. This server rephrases user queries, searches for results using a search tool, and synthesizes answers.
- **server_via_dynamiq.py**: A simplified server for **Stage 2**, which uses the Dynamiq UI to set up nodes, agents, and tools, and leverages an API endpoint to process queries.

---

## Stage 1: Programmatically Building the Search Application

In this stage, we manually define the logic server using Dynamiq's Python library.

### Key Components:

1. **Agents and Tools**:
    - **Rephrasing Agent**: This agent rewrites the user's input query to make it more concise and optimized for search engines.
    - **Search Tool**: This tool uses an external search engine (e.g., SERP Scale) to retrieve relevant information based on the rephrased query.
    - **Answer Synthesizer Agent**: This agent synthesizes an answer from the search results and formats it according to the query.

2. **Workflow**:
    - A workflow is defined using the agents and tools mentioned above. The query first goes through the rephrasing agent, then to the search tool, and finally to the answer synthesizer agent.

3. **Frontend**:
    - A Streamlit-based frontend (`app.py`) allows users to input search queries, and it processes these queries through the defined workflow. The result is streamed back to the user in real time, showing both the sources of information and the final answer.

### How to Run:

Run the Streamlit app:
   ```
   streamlit run app.py
   ```

---

## Stage 2: Building the Application Using Dynamiq UI

In this stage, we utilize the **Dynamiq UI** to visually create the workflow, agents, and tools. Once the workflow is set up, it is deployed as an API, which acts as a backend for the Streamlit app.

### Key Components:

1. **Dynamiq UI**:
    - The agents and tools are created using Dynamiq's graphical interface.
    - The workflow is deployed, and an API endpoint is provided to interact with the workflow.

2. **Backend**:
    - The `server_via_dynamiq.py` file contains a simplified implementation that connects to the Dynamiq API. The queries are sent to this API, and the results are streamed back and displayed to the user in real time.

### How to Run:

1. Set up environment variables for the Dynamiq API:
   ```
   export DYNAMIQ_ENDPOINT=<Your Dynamiq endpoint>
   export DYNAMIQ_API_KEY=<Your Dynamiq API key>
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

---

## Conclusion

This project showcases two flexible approaches to building a search-based application using Dynamiq:

- In **Stage 1**, we manually define the workflow and integrate it into a Python-based backend.
- In **Stage 2**, we leverage the Dynamiq UI to deploy the backend as an API.

Both approaches allow for real-time query processing and answer synthesis, demonstrating the power and versatility of Dynamiq in creating intelligent search tools.

Feel free to experiment with both approaches depending on your needs and preferences!
