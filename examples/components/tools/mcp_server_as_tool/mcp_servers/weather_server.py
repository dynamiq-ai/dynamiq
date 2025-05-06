from datetime import datetime, timedelta, timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

WEATHER_DATA = {"Paris": "Sunny", "Tokyo": "Cloudy", "London": "Windy"}

TIMEZONE_OFFSETS = {"Paris": 2, "Tokyo": 9, "London": 1}


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    weather = WEATHER_DATA.get(location, "Unknown")
    return f"The current weather in {location} is {weather}."


@mcp.tool()
async def get_time(location: str) -> str:
    """Get time for location."""
    offset = TIMEZONE_OFFSETS.get(location, 0)
    utc_now = datetime.now(timezone.utc)
    local_time = (utc_now + timedelta(hours=offset)).strftime("%H:%M")
    return f"The current local time in {location} is {local_time}."


@mcp.tool()
async def list_supported_locations() -> list[str]:
    """
    List all supported locations for weather and time queries.
    """
    return list(WEATHER_DATA.keys())


if __name__ == "__main__":
    mcp.run(transport="sse")
