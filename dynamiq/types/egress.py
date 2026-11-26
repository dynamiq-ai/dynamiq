import re
from urllib.parse import SplitResult, unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_PORTS = {"http": 80, "https": 443}

# A header name as HTTP defines it.
HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")

# Headers that frame a request or tell a front end where to route it, which no policy may allow on an exempt request.
ROUTING_HEADER_NAMES = frozenset(
    {
        "connection",
        "content-length",
        "forwarded",
        "host",
        "keep-alive",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "via",
    }
)
ROUTING_HEADER_PREFIXES = ("x-forwarded-", "x-original-", "x-rewrite-", "x-envoy-", "x-host")


def _origin_and_path(url: str) -> tuple[tuple[str, str, int], str] | None:
    """The scheme, host and port of an http(s) URL without credentials in it, and its path; None for any other."""
    try:
        parts: SplitResult = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS or not parts.hostname or parts.username is not None or parts.password is not None:
        return None
    if port == 0:
        return None
    return (scheme, parts.hostname.rstrip("."), DEFAULT_PORTS[scheme] if port is None else port), parts.path or "/"


def _has_ambiguous_segment(path: str) -> bool:
    """Reports whether a segment of the path could name another path on some server.

    That is a dot segment, one with path parameters (servlet containers drop ";..." and then resolve ".."), a slash or
    backslash hidden by percent-encoding, encoding left after one decoding, which a server that decodes twice resolves,
    and control characters.
    """
    for segment in path.split("/"):
        decoded = unquote(segment)
        if decoded.strip(" \t") in (".", ".."):
            return True
        if any(char in decoded for char in ";%/\\") or any(ord(char) < 0x20 or ord(char) == 0x7F for char in decoded):
            return True
    return False


class EgressPolicy(BaseModel):
    """Limits the HTTP requests a node makes to public IPv4 addresses, and bounds their size and duration.

    The host that runs workflows for others sets it: for every run with `set_default_egress_policy()`, or for one run
    with `RunnableConfig.egress`. The HTTP API call node applies it; without a policy it connects anywhere, as before.

    Attributes:
        exempt_url_prefixes: URLs on the host's private network that its own tools call, such as
            "http://tools.internal/v1/conversations/<id>/". A request skips the address check when its scheme, host
            and port match a prefix and its path is the prefix's path or below it. Every redirect is checked again.
        timeout_seconds: The most one request may take, redirects and reading the body included. It also caps the
            node's own timeout.
        max_response_bytes: The most a response body may hold once decoded.
        max_redirects: The most redirects one request follows.
        exempt_request_headers: Header names an exempt request may carry besides the ones the clients set themselves,
            those that describe the body, "Authorization" and "Cookie", such as "X-Api-Key". Names that frame a request
            or that a front end routes by are refused.
    """

    model_config = ConfigDict(frozen=True)

    exempt_url_prefixes: tuple[str, ...] = ()
    timeout_seconds: float = Field(default=120, gt=0)
    max_response_bytes: int = Field(default=1024 * 1024, gt=0)
    max_redirects: int = Field(default=10, ge=0)
    exempt_request_headers: tuple[str, ...] = ()

    @field_validator("exempt_url_prefixes")
    @classmethod
    def validate_exempt_url_prefixes(cls, prefixes: tuple[str, ...]) -> tuple[str, ...]:
        for prefix in prefixes:
            parts = urlsplit(prefix)
            if _origin_and_path(prefix) is None or parts.query or parts.fragment or _has_ambiguous_segment(parts.path):
                raise ValueError(
                    f"An exempt URL prefix must be an http or https URL with a host and a plain path: {prefix!r}."
                )
        return prefixes

    @field_validator("exempt_request_headers")
    @classmethod
    def validate_exempt_request_headers(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        for name in names:
            lowered = name.lower()
            if (
                not HEADER_NAME.match(name)
                or lowered in ROUTING_HEADER_NAMES
                or lowered.startswith(ROUTING_HEADER_PREFIXES)
            ):
                raise ValueError(f"An exempt request may not be allowed the header {name!r}.")
        return names

    def is_exempt(self, url: str) -> bool:
        """Reports whether a request to url skips the address check."""
        target = _origin_and_path(url)
        if target is None or not self.exempt_url_prefixes:
            return False
        origin, path = target
        if _has_ambiguous_segment(path):
            return False
        for prefix in self.exempt_url_prefixes:
            prefix_origin, prefix_path = _origin_and_path(prefix)
            prefix_path = prefix_path.rstrip("/")
            if origin == prefix_origin and (path == prefix_path or path.startswith(prefix_path + "/")):
                return True
        return False


_default_policy: EgressPolicy | None = None


def set_default_egress_policy(policy: EgressPolicy | None) -> None:
    """Applies policy to every run that does not set `RunnableConfig.egress`; None removes it."""
    global _default_policy
    _default_policy = policy


def get_default_egress_policy() -> EgressPolicy | None:
    return _default_policy
