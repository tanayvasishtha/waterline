"""One HTTP client configuration for every USGS call."""

import httpx

from waterline import __version__

USER_AGENT = f"waterline/{__version__}"
TIMEOUT_S = 60.0
RETRIES = 3


class UsgsError(RuntimeError):
    """A USGS request failed; the message is a single line safe to show a user."""


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(
        transport=transport or httpx.HTTPTransport(retries=RETRIES),
        timeout=TIMEOUT_S,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )


def _describe(error: httpx.HTTPError) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    return type(error).__name__


def _get(client: httpx.Client, url: str, params: dict[str, str] | None) -> httpx.Response:
    try:
        response = client.get(url, params=params)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise UsgsError(f"USGS request failed: {url} ({_describe(error)})") from error
    return response


def get_json(client: httpx.Client, url: str, params: dict[str, str] | None = None) -> object:
    return _get(client, url, params).json()


def get_bytes(client: httpx.Client, url: str) -> bytes:
    return _get(client, url, None).content
