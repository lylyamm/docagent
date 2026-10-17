"""Google Cloud sign-in for Vertex AI, with no key in ``.env``.

The supervisor can run on Gemini through Vertex AI (``AGENT_PROVIDER=vertex``).
Vertex AI takes a short-lived OAuth token, not a fixed key. The token comes from
Google's application default credentials, set once on the machine with::

    gcloud auth application-default login
    gcloud auth application-default set-quota-project <project id>

``google_token()`` returns a function giving a valid token, refreshed when it
expires (about every hour). ``google-auth`` is optional: ``uv sync --extra vertex``.
"""

from collections.abc import Callable

SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def google_token(project: str | None = None) -> Callable[[], str]:
    try:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError
        from google.auth.transport.requests import Request
    except ImportError as exc:
        raise RuntimeError("Vertex AI needs google-auth: run `uv sync --extra vertex`") from exc
    try:
        credentials, _ = google.auth.default(scopes=[SCOPE], quota_project_id=project)
    except DefaultCredentialsError as exc:
        raise RuntimeError(
            "no Google sign-in found: run `gcloud auth application-default login`"
        ) from exc
    request = Request()

    def token() -> str:
        if not credentials.valid:  # never fetched, or expired
            credentials.refresh(request)
        return credentials.token

    return token
