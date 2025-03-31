from datetime import date

from dynamiq import Workflow
from dynamiq.connections import Http as HttpConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.nodes.tools.http_api_call import HttpApiCall, RequestPayloadType
from dynamiq.prompts import Message, Prompt

# API Descriptions
STOCK_HISTORICAL_API_DESCRIPTION = """
Historical Stock Market Data API

Endpoint:
* 'query' (function: str, symbol: str, interval: str)

* 'TIME_SERIES_INTRADAY' - Intraday stock data
    Parameters:
        Required:
            - function='TIME_SERIES_INTRADAY' (str)
            - symbol (str)
            - interval (str: '1min', '5min', '15min', '30min', '60min')
        Optional:
            - month (str: "YYYY-MM")
            - extended_hours (bool)
            - adjusted (bool)

**Request format:**
Provide the function name and parameters in the `params` object:
```json
 {"params": {"function": "TIME_SERIES_INTRADAY", "symbol":"COMPANY", "interval": "INTERVAL", **extra},
 "payload_type": "raw", "data": {}, "url": "", "headers": {}}
```
"""

STOCK_DATA_API_DESCRIPTION = """
Stock Market Data API: Daily, Weekly, Monthly, Real-Time.

Endpoint:
* 'query' (function: str, symbol: str)

**IMPORTANT:** Each request MUST include exactly ONE company symbol (`symbol`).
Multiple symbols in a single request are NOT allowed.

**Function options:**
* 'TIME_SERIES_DAILY' - Daily stock data.
* 'TIME_SERIES_DAILY_ADJUSTED' - Daily stock data with adjustments.
* 'TIME_SERIES_WEEKLY' - Weekly stock data.
* 'TIME_SERIES_WEEKLY_ADJUSTED' - Weekly stock data with adjustments.
* 'TIME_SERIES_MONTHLY' - Monthly stock data.
* 'TIME_SERIES_MONTHLY_ADJUSTED' - Monthly stock data.
* 'REALTIME_BULK_QUOTES' - Bulk real-time quotes for up to 100 US-traded symbols.
* 'REALTIME_OPTIONS' - Real-time US options data.

**Request format:**
```json
 {"params": {"function": "FUNCTION", "symbol":"COMPANY"}, "data": {}, "url": "",
 "payload_type": "raw", "headers": {}}
```
"""

COMPANY_API_DESCRIPTION = """
Financial Data API for Companies.

Endpoint:
* 'query' (function: str, symbol: str)

**Function options:**
* 'OVERVIEW' - Company overview.
* 'ETF_PROFILE' - ETF Profile & Holdings.
* 'DIVIDENDS' - Historical and future dividend distributions.
* 'SPLITS' - Historical split events.
* 'BALANCE_SHEET' - Annual and quarterly balance sheet.
* 'CASH_FLOW' - Annual and quarterly cash flow.
* 'EARNINGS' - Annual and quarterly earnings.

**Request format:**
```json
{"params": {"function": "FUNCTION", "symbol":"COMPANY"}, "data": {}, "url": "",
"payload_type": "raw", "headers": {}}
```
"""

WRITE_REPORT_PROMPT = f"""
Information: "{{{{context}}}}"
---
Based on the above, answer: "{{{{question}}}}" in a detailed, well-structured,
and comprehensive report of at least {{{{word_lower_limit}}}} words.
Provide factual information, numbers if available, and a concrete, well-reasoned analysis.

- Formulate a clear and valid opinion; avoid vague conclusions.
- Use markdown syntax.
- Today's date is {date.today()}.

This report is critical, ensure the highest quality.
"""


def create_api_tools(endpoint_url, api_key):
    """
    Create and initialize API tools for accessing stock market and company financial data.

    Args:
        endpoint_url (str): The base URL for API requests.
        api_key (str): API key for authentication with Alpha Vantage.

    Returns:
        tuple: A tuple containing initialized API tools.
    """
    http_get_connection = HttpConnection(method="GET")

    stock_historical_data_tool = HttpApiCall(
        connection=http_get_connection,
        name="Stock-Historical-API",
        description=STOCK_HISTORICAL_API_DESCRIPTION,
        params={"apikey": api_key},
        url=endpoint_url,
        payload_type=RequestPayloadType.RAW,
    )

    stock_data_tool = HttpApiCall(
        connection=http_get_connection,
        name="Stock-Data-API",
        description=STOCK_DATA_API_DESCRIPTION,
        params={"apikey": api_key},
        url=endpoint_url,
        payload_type=RequestPayloadType.RAW,
    )

    company_data_tool = HttpApiCall(
        connection=http_get_connection,
        name="Company-Data-API",
        description=COMPANY_API_DESCRIPTION,
        params={"apikey": api_key},
        url=endpoint_url,
        payload_type=RequestPayloadType.RAW,
    )

    return stock_data_tool, company_data_tool, stock_historical_data_tool


def initialize_workflow(endpoint_url, api_key):
    """
    Initialize the financial research workflow.

    Args:
        endpoint_url (str): The base URL for API requests.
        api_key (str): API key for authentication with Alpha Vantage.

    Returns:
        Workflow: A configured instance of the financial research workflow.
    """
    openai_connection = OpenAIConnection()

    llm_model = OpenAI(
        connection=openai_connection,
        model="gpt-4o-mini",
        temperature=0.01,
    )

    api_tools = create_api_tools(endpoint_url, api_key)

    financial_research_agent = Agent(
        id="financial_research_agent",
        name="Financial Research Agent",
        role="Generates detailed financial research based on user queries using APIs.",
        llm=llm_model,
        tools=api_tools,
        max_loops=30,
    )

    generate_report_node = OpenAI(
        id="generate_report_node",
        name="generate-report",
        model="gpt-4o-mini",
        connection=OpenAIConnection(),
        prompt=Prompt(messages=[Message(role="user", content=WRITE_REPORT_PROMPT)]),
        temperature=0.35,
        max_tokens=3000,
        input_transformer=InputTransformer(
            selector={
                "context": f"${[financial_research_agent.id]}.output.content",
                "question": "$.input",
                "word_lower_limit": "$.word_lower_limit",
            }
        ),
        depends=[NodeDependency(financial_research_agent)],
    )

    workflow = Workflow()
    workflow.flow.add_nodes(financial_research_agent)
    workflow.flow.add_nodes(generate_report_node)
    return workflow


if __name__ == "__main__":
    api_key = "YOUR_API_KEY_AlphaVantage"  # https://www.alphavantage.co/
    url = "https://www.alphavantage.co/query"

    wf = initialize_workflow(url, api_key)

    possible_questions_to_ask = [
        "NVIDIA stocks vs AMD stocks",
        "Provide a detailed financial analysis of Apple,including its recent stock performance, "
        "historical price trends over the past year, and key company financials.",
        "Which stocks of MAANG companies were the most successful during the last month?",
    ]

    result = wf.run(input_data={"input": possible_questions_to_ask[1], "word_lower_limit": 300})

    print("Result:")
    print(result.output.get("generate_report_node", {}).get("output", {}).get("content"))
