from dynamiq.connections import Http as HttpConnection
from dynamiq.nodes.tools.http_api_call import HttpApiCall, ResponseType


def main():
    # Create an HTTP connection
    connection = HttpConnection(
        method="GET",
        url="https://catfact.ninja/fact",
    )

    # Create an instance of HttpApiCall
    api_call = HttpApiCall(
        connection=connection,
        success_codes=[200, 201],
        timeout=60,
        response_type=ResponseType.JSON,
        params={"limit": 10},
    )

    # Prepare input data
    input_data = {}

    # Run the API call
    try:
        result = api_call.run(input_data)
        print(result.output)
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
