from dynamiq.connections import Coda
from dynamiq.memory import Memory
from dynamiq.memory.backends.in_memory import InMemory
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.coda.docs import CodaCreateDoc, CodaGetDocInfo, CodaListAvailableDocs
from dynamiq.nodes.tools.coda.pages import CodaCreatePage, CodaListPages
from dynamiq.nodes.tools.coda.tables import (
    CodaGetTable,
    CodaGetTableColumn,
    CodaGetTableRows,
    CodaInsertUpsertRows,
    CodaListTableColumns,
    CodaListTableRows,
    CodaListTables,
    CodaUpdateRow,
)
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

if __name__ == "__main__":
    coda_connection = Coda(api_key="")
    coda_list_docs_node = CodaListAvailableDocs(connection=coda_connection, is_optimized_for_agents=False)
    coda_get_doc_info_node = CodaGetDocInfo(connection=coda_connection, is_optimized_for_agents=False)
    coda_create_doc_node = CodaCreateDoc(connection=coda_connection, is_optimized_for_agents=False)
    coda_create_page_node = CodaCreatePage(connection=coda_connection, is_optimized_for_agents=False)
    coda_list_pages_node = CodaListPages(connection=coda_connection, is_optimized_for_agents=False)
    coda_list_tables_node = CodaListTables(connection=coda_connection, is_optimized_for_agents=False)
    coda_get_table_node = CodaGetTable(connection=coda_connection, is_optimized_for_agents=False)
    coda_list_table_rows_node = CodaListTableRows(connection=coda_connection, is_optimized_for_agents=False)
    coda_update_row_node = CodaUpdateRow(connection=coda_connection, is_optimized_for_agents=False)
    coda_insert_upsert_rows_node = CodaInsertUpsertRows(connection=coda_connection, is_optimized_for_agents=False)
    coda_list_table_columns_node = CodaListTableColumns(connection=coda_connection, is_optimized_for_agents=False)
    coda_get_table_column_node = CodaGetTableColumn(connection=coda_connection, is_optimized_for_agents=False)
    coda_get_table_rows_node = CodaGetTableRows(connection=coda_connection, is_optimized_for_agents=False)

    memory = Memory(backend=InMemory())

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0)
    agent = ReActAgent(
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
