"""Turn an auth/lookup status into a message that says what is actually wrong."""

from dynamiq.cli.config import Settings


def mask_token(token: str | None) -> str:
    """Show just enough of a token to tell two apart, never the whole thing."""
    if not token:
        return "<not set>"
    if len(token) < 12:
        return "****"
    return token[:4] + "..."


def token_rejected(settings: Settings) -> str:
    return (
        f"Token rejected (401) by {settings.api_host}. The token {mask_token(settings.api_key)} comes from "
        f"{settings.source_of('api_key')}; check it or run `dynamiq config`."
    )


def access_failure(settings: Settings, kind: str, resource_id: str, status: int) -> str:
    """Why `kind` `resource_id` could not be read, by HTTP status.

    A 403 and a 404 are different problems: the first means the id is right and this token
    may not see it, the second that no such id exists. Reporting both as "not found" sends
    people hunting for a typo when the fix is a different token.
    """
    if status == 401:
        return token_rejected(settings)
    if status == 403:
        return f"{kind} {resource_id} is not accessible with this token (403)."
    if status == 404:
        return f"{kind} {resource_id} was not found (404)."
    return f"Could not read {kind} {resource_id}: HTTP {status}."
