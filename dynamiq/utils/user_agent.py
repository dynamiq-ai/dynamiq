import importlib.metadata

try:
    VERSION = importlib.metadata.version("dynamiq")
except importlib.metadata.PackageNotFoundError:
    VERSION = "unknown"

# The platform tells the SDK and the CLI apart from other API clients by the User-Agent.
USER_AGENT = f"dynamiq-python/{VERSION}"
CLI_USER_AGENT = f"dynamiq-cli/{VERSION}"
