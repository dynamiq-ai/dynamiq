from datetime import datetime


def format_duration(start: datetime, end: datetime) -> str:
    """
    Format the duration between two datetime objects into a human-readable string.

    This function calculates the time difference between the start and end datetimes and
    returns a formatted string representing the duration in milliseconds, seconds, minutes,
    or hours, depending on the length of the duration.

    Args:
        start (datetime): The starting datetime.
        end (datetime): The ending datetime.

    Returns:
        str: A formatted string representing the duration.
             - For durations less than 1 second: "Xms" (milliseconds)
             - For durations between 1 second and 1 minute: "Xs" (seconds)
             - For durations between 1 minute and 1 hour: "Xm" (minutes)
             - For durations of 1 hour or more: "Xh" (hours)

    Examples:
        >>> from datetime import datetime, timedelta
        >>> start = datetime(2023, 1, 1, 12, 0, 0)
        >>> print(format_duration(start, start + timedelta(milliseconds=500)))
        500ms
        >>> print(format_duration(start, start + timedelta(seconds=45)))
        45.0s
        >>> print(format_duration(start, start + timedelta(minutes=30)))
        30.0m
        >>> print(format_duration(start, start + timedelta(hours=2)))
        2.0h
    """
    delta = end - start
    total_seconds = delta.total_seconds()

    if total_seconds < 1:
        return f"{total_seconds * 1000:.0f}ms"
    elif total_seconds < 60:
        return f"{round(total_seconds, 1)}s"
    elif total_seconds < 3600:
        return f"{round(total_seconds / 60, 1)}m"
    else:
        return f"{round(total_seconds / 3600, 1)}h"
