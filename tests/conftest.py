import pytest

from docagent import observability


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    """No test sends traces, even with Langfuse keys in a developer's .env;
    tests/test_observability.py turns it back on with a fake client."""
    monkeypatch.setattr(observability, "enabled", lambda settings=None: False)
    monkeypatch.setattr(observability, "_client", None)
