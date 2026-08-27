"""Authentication helpers for DataHub's internal OpenAI-compatible gateway."""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import urlsplit


_SERVICE_TOKEN_ENV_KEYS = (
    "AI_SERVICE_INTERNAL_TOKEN",
    "DATAHUB_INTERNAL_SERVICE_TOKEN",
    "INTERNAL_SERVICE_TOKEN",
    "LIGHTRAG_API_KEY",
)


def resolve_datahub_internal_openai_api_key(
    *,
    binding: str | None,
    base_url: str | None,
    configured_api_key: str | None,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Choose the service token only for DataHub's private OpenAI gateway.

    The runtime normally treats ``*_LLM_BINDING_API_KEY`` as a provider key.
    DataHub routes selected roles through ai-service instead, whose gateway
    requires the service-to-service token. Restricting this substitution to an
    exact internal origin and path prevents an internal token from ever being
    sent to an external model endpoint.
    """
    values = os.environ if environ is None else environ
    if not _is_datahub_internal_openai_endpoint(binding, base_url, values):
        return configured_api_key
    for key in _SERVICE_TOKEN_ENV_KEYS:
        token = str(values.get(key) or "").strip()
        if token:
            return token
    return configured_api_key


def _is_datahub_internal_openai_endpoint(
    binding: str | None,
    base_url: str | None,
    environ: Mapping[str, str],
) -> bool:
    if str(binding or "").strip().lower() != "openai":
        return False
    ai_service_url = str(environ.get("AI_SERVICE_URL") or "").strip()
    candidate_url = str(base_url or "").strip()
    if not ai_service_url or not candidate_url:
        return False
    try:
        service = urlsplit(ai_service_url)
        candidate = urlsplit(candidate_url)
    except ValueError:
        return False
    if not _same_origin(service, candidate):
        return False
    prefix = str(environ.get("AI_SERVICE_INTERNAL_PREFIX") or "/ai/internal")
    prefix = prefix.strip().strip("/")
    expected_path = f"/{prefix}/openai/v1" if prefix else "/openai/v1"
    normalized_path = candidate.path.rstrip("/")
    return normalized_path == expected_path or normalized_path.startswith(
        expected_path + "/"
    )


def _same_origin(left, right) -> bool:
    if left.scheme.lower() not in {"http", "https"}:
        return False
    if right.scheme.lower() != left.scheme.lower():
        return False
    if not left.hostname or not right.hostname:
        return False
    if left.hostname.lower() != right.hostname.lower():
        return False
    try:
        return _normalized_port(left) == _normalized_port(right)
    except ValueError:
        return False


def _normalized_port(parsed) -> int:
    if parsed.port is not None:
        return parsed.port
    return 443 if parsed.scheme.lower() == "https" else 80
