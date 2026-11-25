from urllib.parse import SplitResult, unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_PORTS = {"http": 80, "https": 443}


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
    return (scheme, parts.hostname.rstrip("."), port or DEFAULT_PORTS[scheme]), parts.path or "/"


def _has_ambiguous_segment(path: str) -> bool:
    """Whether a segment of the path is a dot segment or hides a slash, which a server may resolve differently."""
    for segment in path.split("/"):
        decoded = unquote(segment)
        if decoded in (".", "..") or "/" in decoded or "\\" in decoded:
            return True
    return False


class EgressPolicy(BaseModel):
    """Limits the HTTP requests a node makes to public IPv4 addresses, and bounds their size and duration.

    The host that runs workflows for others sets it: for every run with `set_default_egress_policy()`, or for one run
    with `RunnableConfig.egress`. The HTTP API call node applies it; without a policy it connects anywhere, as before.

    Attributes:
        exempt_url_prefixes: URLs on the host's private network that its own tools call, such as
            "http://catalyst.internal/v1/conversations/<id>/". A request skips the address check when its scheme, host
            and port match a prefix and its path is the prefix's path or below it. Every redirect is checked again.
        timeout_seconds: The most one request may take, redirects and reading the body included. It also caps the
            node's own timeout.
        max_response_bytes: The most a response body may hold once decoded.
        max_redirects: The most redirects one request follows.
    """

    model_config = ConfigDict(frozen=True)

    exempt_url_prefixes: tuple[str, ...] = ()
    timeout_seconds: float = Field(default=120, gt=0)
    max_response_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_redirects: int = Field(default=10, ge=0)

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
