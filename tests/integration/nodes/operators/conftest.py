from datetime import date


def calendar_reason(year: int, month: int, day: int) -> str:
    """Python's reason for refusing a day the calendar lacks, which Python 3.14 reworded."""
    try:
        date(year, month, day)
    except ValueError as e:
        return str(e)
    raise AssertionError(f"{year}-{month}-{day} is a day of the calendar")
