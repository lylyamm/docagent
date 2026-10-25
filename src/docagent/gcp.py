"""Google Cloud sign-in for Vertex AI, with no key in ``.env``.

The supervisor can run on Gemini through Vertex AI (``AGENT_PROVIDER=vertex``).
Vertex AI takes a short-lived OAuth token, not a fixed key. The token comes from
Google's application default credentials, set once on the machine with::

    gcloud auth application-default login
    gcloud auth application-default set-quota-project <project id>

``google_token()`` returns a function giving a valid token, refreshed when it
expires (about every hour). ``google-auth`` is optional: ``uv sync --extra vertex``.

``identity_token()`` is for the front on Cloud Run: the API is private, so each call
carries a token proving who calls (the front's service account).
"""

from collections.abc import Callable

import httpx

SCOPE = "https://www.googleapis.com/auth/cloud-platform"
# Inside Google Cloud only: the metadata server signs tokens for the service's account.
METADATA_IDENTITY = (
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity"
)


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


def identity_token(audience: str, client: httpx.Client | None = None) -> str:
    """An identity token for ``audience``, the URL of a private Cloud Run service.

    Asked to the metadata server, so it works only on Google Cloud; valid one hour.
    Raises ``httpx.HTTPError``.
    """
    get = client.get if client else httpx.get
    response = get(
        METADATA_IDENTITY,
        params={"audience": audience},
        headers={"Metadata-Flavor": "Google"},
        timeout=5,
    )
    response.raise_for_status()
    return response.text
