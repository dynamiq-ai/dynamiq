from io import BytesIO

from dynamiq.nodes.transformers.csv import CSVToList, CSVToListTransformerInputSchema


def parse_csv(value, is_header_exists: bool = True):
    return CSVToList().execute(
        CSVToListTransformerInputSchema(value=value, is_header_exists=is_header_exists)
    )["content"]


def test_bytesio_with_utf8_bom_keeps_the_first_header():
    """Excel's "CSV UTF-8" export starts with a BOM; the first header must not keep it."""
    rows = parse_csv(BytesIO("\ufeffname,age\nAlice,30\nBob,25\n".encode()))

    assert rows == [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]


def test_string_with_utf8_bom_keeps_the_first_header():
    rows = parse_csv("\ufeffname,age\nAlice,30\n")

    assert rows == [{"name": "Alice", "age": "30"}]


def test_bytesio_with_utf8_bom_without_header_keeps_the_first_cell():
    rows = parse_csv(BytesIO("\ufeffname,age\nAlice,30\n".encode()), is_header_exists=False)

    assert rows == [{"column0": "name", "column1": "age"}, {"column0": "Alice", "column1": "30"}]


def test_csv_without_bom_is_unchanged():
    rows = parse_csv(BytesIO(b"name,age\nAlice,30\n"))

    assert rows == [{"name": "Alice", "age": "30"}]
