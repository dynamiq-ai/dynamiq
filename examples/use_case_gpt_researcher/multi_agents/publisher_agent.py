def run_publisher(context: dict, **kwargs) -> dict:
    """Generates a formatted research report based on provided context data."""

    report_date = context.get("date")
    introduction = context.get("introduction")
    table_of_contents = context.get("table_of_contents")
    conclusion = context.get("conclusion")

    headers = context.get("headers")
    title = headers.get("title")
    date_label = headers.get("date")

    references = "\n".join(context.get("sources", []))
    sections = "\n\n".join(context.get("research_data", []))

    # Construct the report layout
    report_layout = f"""# {title}
#### {date_label}: {report_date}

## {headers.get("introduction", "Introduction")}
{introduction}

## {headers.get("table_of_contents", "Table of Contents")}
{table_of_contents}

{sections}

## {headers.get("conclusion", "Conclusion")}
{conclusion}

## {headers.get("references", "References")}
{references}
"""

    return {"report": report_layout, "result": "success"}
