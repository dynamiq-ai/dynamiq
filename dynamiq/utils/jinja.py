from functools import lru_cache

from jinja2 import Environment, Template, meta

# Prompts render the same few templates on every LLM call; distinct sources beyond this are evicted, oldest first.
TEMPLATES_CACHE_SIZE = 512

_PARSING_ENVIRONMENT = Environment(autoescape=True)


@lru_cache(maxsize=TEMPLATES_CACHE_SIZE)
def compile_template(source: str) -> Template:
    """
    Compile a template as `jinja2.Template(source)` does, reusing the result for a source compiled before.

    Compiling costs far more CPU than rendering, and prompts render the same few templates on every
    LLM call. A compiled template keeps no state between renders, so threads share it. A source that
    fails to compile is not cached and raises every time.

    Args:
        source (str): The template source.

    Returns:
        Template: The compiled template.
    """
    return Template(source)


@lru_cache(maxsize=TEMPLATES_CACHE_SIZE)
def template_parameters(source: str) -> frozenset[str]:
    """
    Find the variables a template reads without defining them, reusing the result for a source seen before.

    Args:
        source (str): The template source.

    Returns:
        frozenset[str]: The names of the undeclared variables.
    """
    return frozenset(meta.find_undeclared_variables(_PARSING_ENVIRONMENT.parse(source)))
