from typing import Any

from dynamiq.connections import GoogleOAuth2
from dynamiq.nodes.tools.google_sheets import (
    BatchGetValues,
    BatchUpdateValues,
    ClearValues,
    CreateSpreadsheet,
    CreateSpreadsheetColumn,
    CreateSpreadsheetFromJSON,
    CreateSpreadsheetRow,
    FindWorksheetByTitle,
    FormatCellRange,
    GetSheetNames,
    GetSpreadsheetInfo,
    LookupSpreadsheetRow,
)


def create_spreadsheet(title: str):
    """Creates a new Google Sheets spreadsheet."""
    tool = CreateSpreadsheet(connection=GoogleOAuth2())
    result = tool.run(input_data={"title": title or "Test Spreadsheet"})
    tool.close()
    return result.output


def update_spreadsheet(
    spreadsheet_id: str,
    sheet_name: str,
    values: list[list[Any]],
    first_cell_location: str | None = None,
    value_input_option: str | None = None,
    include_values_in_response: bool | None = None,
):
    """Updates values in a specified range of cells in a Google Sheets spreadsheet."""
    tool = BatchUpdateValues(connection=GoogleOAuth2())
    tool.run(
        input_data={
            "spreadsheet_id": spreadsheet_id,
            "sheet_name": sheet_name or "Sheet1",
            "values": values or [[1, 2], [3, 4]],
            "first_cell_location": first_cell_location,
            "value_input_option": value_input_option or "USER_ENTERED",
            "include_values_in_response": include_values_in_response,
        }
    )
    tool.close()


def read_spreadsheet(spreadsheet_id: str, ranges: list[str]):
    """Reads values from a specified range of cells in a Google Sheets spreadsheet."""
    tool = BatchGetValues(connection=GoogleOAuth2())
    tool.run(input_data={"spreadsheet_id": spreadsheet_id, "ranges": ranges or ["A1:B2"]})
    tool.close()


def find_worksheet_by_title(spreadsheet_id: str, title: str):
    """Finds a worksheet by title in a Google Sheets spreadsheet."""
    tool = FindWorksheetByTitle(connection=GoogleOAuth2())
    tool.run(input_data={"spreadsheet_id": spreadsheet_id, "title": title or "Sheet1"})
    tool.close()


def lookup_spreadsheet_row(spreadsheet_id: str, query: str):
    """Looks up a row in a Google Sheets spreadsheet."""
    tool = LookupSpreadsheetRow(connection=GoogleOAuth2())
    tool.run(
        input_data={
            "spreadsheet_id": spreadsheet_id,
            "query": query,
        }
    )
    tool.close()


def get_spreadsheet_info(spreadsheet_id: str):
    """Gets the metadata of a Google Sheets spreadsheet."""
    tool = GetSpreadsheetInfo(connection=GoogleOAuth2())
    tool.run(input_data={"spreadsheet_id": spreadsheet_id})
    tool.close()


def get_sheet_names(spreadsheet_id: str):
    """Gets the names of the sheets in a Google Sheets spreadsheet."""
    tool = GetSheetNames(connection=GoogleOAuth2())
    tool.run(input_data={"spreadsheet_id": spreadsheet_id})
    tool.close()


def clear_values_in_a_range(spreadsheet_id: str, range: str):
    """Clears values in a specified range of cells in a Google Sheets spreadsheet."""
    tool = ClearValues(connection=GoogleOAuth2())
    tool.run(input_data={"spreadsheet_id": spreadsheet_id, "range": range or "A1:B2"})
    tool.close()


def create_sheet_from_json(title: str, sheet_name: str, json_data: list[dict[str, Any]]):
    """Creates a new sheet in a Google Sheets spreadsheet from a JSON object."""
    tool = CreateSpreadsheetFromJSON(connection=GoogleOAuth2())
    tool.run(input_data={"title": title, "sheet_name": sheet_name, "json_data": json_data})
    tool.close()


def create_sheet_row(spreadsheet_id: str, sheet_id: int, row_index: int):
    """Creates a new row in a worksheet in a Google Sheets spreadsheet."""
    tool = CreateSpreadsheetRow(connection=GoogleOAuth2())
    tool.run(input_data={"spreadsheet_id": spreadsheet_id, "sheet_id": sheet_id, "insert_index": row_index})
    tool.close()


def create_sheet_column(spreadsheet_id: str, sheet_id: int, column_index: int):
    """Creates a new column in a worksheet in a Google Sheets spreadsheet."""
    tool = CreateSpreadsheetColumn(connection=GoogleOAuth2())
    tool.run(input_data={"spreadsheet_id": spreadsheet_id, "sheet_id": sheet_id, "insert_index": column_index})
    tool.close()


def format_cell_range(
    spreadsheet_id: str,
    worksheet_id: int,
    start_row_index: int,
    end_row_index: int,
    start_column_index: int,
    end_column_index: int,
    bold: bool,
    italic: bool,
    underline: bool,
    strikethrough: bool,
    font_size: int,
    red: float,
    green: float,
    blue: float,
):
    """Formats a cell range in a Google Sheets spreadsheet."""
    tool = FormatCellRange(connection=GoogleOAuth2())
    tool.run(
        input_data={
            "spreadsheet_id": spreadsheet_id,
            "worksheet_id": worksheet_id,
            "start_row_index": start_row_index,
            "end_row_index": end_row_index,
            "start_column_index": start_column_index,
            "end_column_index": end_column_index,
            "bold": bold,
            "italic": italic,
            "underline": underline,
            "strikethrough": strikethrough,
            "font_size": font_size,
            "red": red,
            "green": green,
            "blue": blue,
        }
    )
    tool.close()


if __name__ == "__main__":
    create_spreadsheet_result = create_spreadsheet("Test Spreadsheet")
    spreadsheet_id = create_spreadsheet_result["content"]

    update_spreadsheet(spreadsheet_id, "Sheet1", [["John", 1], ["Jane", 2]])
    update_spreadsheet(spreadsheet_id, "Sheet1", [[1, 2], [3, 4]], "D4")

    get_spreadsheet_info(spreadsheet_id)
    get_sheet_names(spreadsheet_id)
    find_worksheet_by_title(spreadsheet_id, "Sheet1")
    lookup_spreadsheet_row(spreadsheet_id, "John")

    read_spreadsheet(spreadsheet_id, ["A1:B2"])
    find_worksheet_by_title(spreadsheet_id, "Sheet1")
    format_cell_range(spreadsheet_id, 0, 3, 6, 3, 6, False, False, False, False, 11, 0.5, 0.5, 0.5)
    clear_values_in_a_range(spreadsheet_id, "D4:F6")

    json_data = [
        {"Name": "John", "Age": 25, "City": "New York"},
        {"Name": "Jane", "Age": 30, "City": "Los Angeles"},
        {"Name": "Jim", "Age": 35, "City": "Chicago"},
    ]

    create_sheet_from_json("Test Spreadsheet from JSON", "Test Sheet from JSON", json_data)

    create_sheet_row(spreadsheet_id, 0, 0)
    create_sheet_column(spreadsheet_id, 0, 0)
