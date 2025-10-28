from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AuthSchemeType(str, Enum):
    API_KEY = "api_key"
    HTTP = "http"
    OAUTH2 = "oauth2"
    OPEN_ID_CONNECT = "open_id_connect"
    SERVICE_ACCOUNT = "service_account"


class AuthSchemeLocation(str, Enum):
    HEADER = "header"
    QUERY = "query"
    COOKIE = "cookie"


class AuthCredentialType(str, Enum):
    API_KEY = "api_key"
    HTTP = "http"
    OAUTH2 = "oauth2"
    OPEN_ID_CONNECT = "open_id_connect"
    SERVICE_ACCOUNT = "service_account"


class AuthScheme(BaseModel):
    """
    Describes how an external API expects credentials.
    Mirrors OpenAPI-style security schemes in a simplified form.
    """

    type: AuthSchemeType
    name: str | None = None
    description: str | None = None
    location: AuthSchemeLocation | None = Field(default=None, alias="in")
    bearer_format: str | None = None
    token_url: str | None = None
    scopes: dict[str, str] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class AuthCredential(BaseModel):
    """
    Represents provided or exchanged credential material.
    """

    type: AuthCredentialType
    api_key: str | None = None
    token: str | None = None
    refresh_token: str | None = None
    username: str | None = None
    password: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    scopes: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class AuthConfig(BaseModel):
    """
    Bundles an auth scheme with credential data (initial or exchanged).
    """

    scheme: AuthScheme
    credential: AuthCredential | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_tool_payload(self) -> dict[str, Any]:
        """
        Produce a helper payload that tools can consume directly.

        Returns:
            dict[str, Any]: A dictionary containing resolved headers/query params plus raw config.
        """
        payload: dict[str, Any] = {
            "raw_config": self.model_dump(mode="json"),
            "headers": {},
            "query": {},
            "cookies": {},
            "metadata": self.metadata.copy(),
        }

        scheme = self.scheme
        credential = self.credential

        if scheme.type == AuthSchemeType.API_KEY:
            key_name = scheme.name or credential.extra.get("key_name") if credential else None
            key_name = key_name or "api_key"
            value = credential.api_key if credential else None
            location = scheme.location or AuthSchemeLocation.HEADER
            if value:
                if location == AuthSchemeLocation.HEADER:
                    payload["headers"][key_name] = value
                elif location == AuthSchemeLocation.QUERY:
                    payload["query"][key_name] = value
                elif location == AuthSchemeLocation.COOKIE:
                    payload["cookies"][key_name] = value
        elif scheme.type == AuthSchemeType.HTTP:
            if credential and credential.token:
                auth_header = credential.token
                if scheme.bearer_format or credential.extra.get("bearer", True):
                    auth_header = f"Bearer {credential.token}"
                payload["headers"]["Authorization"] = auth_header
            elif credential and credential.username and credential.password:
                payload["metadata"]["basic_auth"] = {
                    "username": credential.username,
                    "password": credential.password,
                }
        elif scheme.type in {AuthSchemeType.OAUTH2, AuthSchemeType.OPEN_ID_CONNECT, AuthSchemeType.SERVICE_ACCOUNT}:
            if credential and credential.token:
                payload["headers"]["Authorization"] = f"Bearer {credential.token}"
                if credential.refresh_token:
                    payload["metadata"]["refresh_token"] = credential.refresh_token
            if scheme.token_url:
                payload["metadata"]["token_url"] = scheme.token_url
            if scheme.scopes:
                payload["metadata"]["scopes"] = list(scheme.scopes.keys())

        return payload


class AuthRequest(BaseModel):
    """
    Represents an authentication request emitted by a tool.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    tool_id: str | None = None
    tool_name: str | None = None
    required: AuthConfig | dict[str, Any] | None = None
    message: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)
