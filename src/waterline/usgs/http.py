"""One HTTP client configuration for every USGS call.

Both the NIMS and OGC APIs are keyed by api.data.gov. Anonymous requests share a low
hourly cap; a free key (https://apiwaterdata.usgs.gov/signup) raises it. The key comes
from the WATERLINE_USGS_API_KEY environment variable and is never written to the repo.
"""

import os
import time

import httpx

from waterline import __version__

USER_AGENT = f"waterline/{__version__}"
TIMEOUT_S = 60.0
RETRIES = 3
RATE_LIMIT_RETRIES = 3
MIN_REQUEST_GAP_S = 0.35
_last_request = 0.0


class UsgsError(RuntimeError):
    """A USGS request failed; the message is a single line safe to show a user."""


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    headers = {"User-Agent": USER_AGENT}
    if key := os.environ.get("WATERLINE_USGS_API_KEY"):
        headers["X-Api-Key"] = key
    return httpx.Client(
        transport=transport or httpx.HTTPTransport(retries=RETRIES),
        timeout=TIMEOUT_S,
        headers=headers,
        follow_redirects=True,
    )


def _describe(error: httpx.HTTPError) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    return type(error).__name__


def _throttle() -> None:
    global _last_request
    gap = MIN_REQUEST_GAP_S - (time.monotonic() - _last_request)
    if gap > 0:
        time.sleep(gap)
    _last_request = time.monotonic()


def _get(client: httpx.Client, url: str, params: dict[str, str] | None) -> httpx.Response:
    for attempt in range(RATE_LIMIT_RETRIES):
        _throttle()
        try:
            response = client.get(url, params=params)
        except httpx.HTTPError as error:
            raise UsgsError(f"USGS request failed: {url} ({_describe(error)})") from error
        if response.status_code != 429:
            try:
                response.raise_for_status()
            except httpx.HTTPError as error:
                raise UsgsError(f"USGS request failed: {url} ({_describe(error)})") from error
            return response
        if attempt == RATE_LIMIT_RETRIES - 1:
            raise UsgsError(f"USGS rate limited after retries: {url}")
        wait = float(response.headers.get("Retry-After", 2 * (attempt + 1)))
        time.sleep(min(wait, 30.0))
    raise UsgsError(f"USGS rate limited after retries: {url}")


def get_json(client: httpx.Client, url: str, params: dict[str, str] | None = None) -> object:
    return _get(client, url, params).json()


def get_bytes(client: httpx.Client, url: str) -> bytes:
    return _get(client, url, None).content
