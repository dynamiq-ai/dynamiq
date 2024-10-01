import textwrap


def _validate_input_data(input_data):
    """
    Validate the input data dictionary for required keys.

    Args:
        input_data (dict): The input data dictionary to validate

    Raises:
        KeyError: If any required key is missing from the input_data dictionary
    """
    required_keys = ["dates", "location", "cities", "interests"]
    for key in required_keys:
        if key not in input_data:
            raise KeyError(f"Missing required key: {key}")


def _format_prompt(template, input_data):
    """
    Format the prompt template with input data.

    Args:
        template (str): The prompt template string
        input_data (dict): The input data dictionary

    Returns:
        str: The formatted prompt string
    """
    return textwrap.dedent(template).format(**input_data)


def generate_customer_prompt(input_data):
    """
    Generate a detailed customer prompt for comprehensive trip planning.

    This function creates a prompt that instructs to analyze and select the best city,
    compile an in-depth city guide, and create a 7-day travel itinerary.

    Args:
        input_data (dict): A dictionary containing trip information with the following keys:
            - dates (str): The dates of the trip
            - location (str): The traveler's starting location
            - cities (str): A list of potential city options
            - interests (str): The traveler's interests

    Returns:
        str: A formatted string containing the detailed customer prompt

    Raises:
        KeyError: If any required key is missing from the input_data dictionary
    """
    _validate_input_data(input_data)

    template = """
    Analyze and select the best city for the trip based on weather patterns, seasonal events, and travel costs.
    Compare multiple cities considering current weather, upcoming events, and travel expenses. Provide a detailed
    report on the chosen city, including flight costs, weather forecast, and attractions.

    Next, compile an in-depth city guide with key attractions, local customs, events, and daily activity recommendations.
    Include hidden gems, cultural hotspots, must-visit landmarks, weather forecasts, and costs. The guide should be rich
    in cultural insights and practical tips to enhance the travel experience.

    Finally, expand the guide into a 7-day travel itinerary with detailed daily plans, including weather forecasts,
    places to eat, packing suggestions, and a budget breakdown. Suggest actual places to visit, hotels, and restaurants.
    The itinerary should cover all aspects of the trip, from arrival to departure, with a daily schedule, recommended
    clothing, items to pack, and a detailed budget.

    Trip Date: {dates}
    Traveling from: {location}
    City Options: {cities}
    Traveler Interests: {interests}
    """  # noqa: E501

    return _format_prompt(template, input_data)


def generate_simple_customer_prompt(input_data):
    """
    Generate a simplified customer prompt for basic trip planning.

    This function creates a prompt that focuses on providing a city guide based on key factors.

    Args:
        input_data (dict): A dictionary containing trip information with the following keys:
            - dates (str): The dates of the trip
            - location (str): The traveler's starting location
            - cities (str): A list of potential city options
            - interests (str): The traveler's interests

    Returns:
        str: A formatted string containing the simplified customer prompt

    Raises:
        KeyError: If any required key is missing from the input_data dictionary
    """
    _validate_input_data(input_data)

    template = """
    Provide the best city guide for the trip based on weather patterns, seasonal events, and travel costs.
    Compare multiple cities considering current weather, upcoming events, and travel expenses. Provide a
    detailed report on the chosen city, including flight costs, weather forecast, and attractions.

    Trip Date: {dates}
    Traveling from: {location}
    City Options: {cities}
    Traveler Interests: {interests}
    """

    return _format_prompt(template, input_data)
