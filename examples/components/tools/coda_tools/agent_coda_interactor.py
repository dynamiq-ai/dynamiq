from dynamiq.memory import Memory
from dynamiq.memory.backends.in_memory import InMemory
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.types import InferenceMode

from dynamiq.connections import Coda
from dynamiq.nodes.tools.coda.docs import CreateDoc, GetDocInfo, ListAvailableDocs
from dynamiq.nodes.tools.coda.pages import CreatePage, ListPages
from dynamiq.nodes.tools.coda.tables import (
    GetTable,
    GetTableColumn,
    GetTableRows,
    InsertUpsertRows,
    ListTableColumns,
    ListTableRows,
    ListTables,
    UpdateRow,
)
from examples.utils import setup_llm

if __name__ == "__main__":
    coda_connection = Coda()
    coda_list_docs_node = ListAvailableDocs(connection=coda_connection, is_optimized_for_agents=False)
    coda_get_doc_info_node = GetDocInfo(connection=coda_connection, is_optimized_for_agents=False)
    coda_create_doc_node = CreateDoc(connection=coda_connection, is_optimized_for_agents=False)
    coda_create_page_node = CreatePage(connection=coda_connection, is_optimized_for_agents=False)
    coda_list_pages_node = ListPages(connection=coda_connection, is_optimized_for_agents=False)
    coda_list_tables_node = ListTables(connection=coda_connection, is_optimized_for_agents=False)
    coda_get_table_node = GetTable(connection=coda_connection, is_optimized_for_agents=False)
    coda_list_table_rows_node = ListTableRows(connection=coda_connection, is_optimized_for_agents=False)
    coda_update_row_node = UpdateRow(connection=coda_connection, is_optimized_for_agents=False)
    coda_insert_upsert_rows_node = InsertUpsertRows(connection=coda_connection, is_optimized_for_agents=False)
    coda_list_table_columns_node = ListTableColumns(connection=coda_connection, is_optimized_for_agents=False)
    coda_get_table_column_node = GetTableColumn(connection=coda_connection, is_optimized_for_agents=False)
    coda_get_table_rows_node = GetTableRows(connection=coda_connection, is_optimized_for_agents=False)

    memory = Memory(backend=InMemory())

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0)
    agent = Agent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[
            coda_list_docs_node,
            coda_get_doc_info_node,
            coda_create_doc_node,
            coda_create_page_node,
            coda_list_pages_node,
            coda_list_tables_node,
            coda_get_table_node,
            coda_list_table_rows_node,
            coda_update_row_node,
            coda_insert_upsert_rows_node,
            coda_list_table_columns_node,
            coda_get_table_column_node,
            coda_get_table_rows_node,
        ],
        inference_mode=InferenceMode.XML,
        memory=memory,
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = agent.run({"input": user_input})
        response_content = response.output.get("content")
        print(f"AI: {response_content}")
