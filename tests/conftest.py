import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


Route = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def mock_client() -> Callable[[Route], httpx.Client]:
    def build(handler: Route) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))

    return build
