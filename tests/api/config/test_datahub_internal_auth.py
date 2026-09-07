"""Regression coverage for the DataHub internal OpenAI gateway token bridge."""

from lightrag.api.datahub_internal_auth import (
    resolve_datahub_internal_openai_api_key,
)


def _environment(**overrides: str) -> dict[str, str]:
    values = {
        "AI_SERVICE_URL": "http://ai-service:8085",
        "AI_SERVICE_INTERNAL_PREFIX": "/ai/internal",
        "LIGHTRAG_API_KEY": "runtime-token",
    }
    values.update(overrides)
    return values


def test_internal_openai_gateway_uses_dedicated_service_token():
    result = resolve_datahub_internal_openai_api_key(
        binding="openai",
        base_url="http://ai-service:8085/ai/internal/openai/v1/vlm",
        configured_api_key="provider-token",
        environ=_environment(AI_SERVICE_INTERNAL_TOKEN="dedicated-token"),
    )

    assert result == "dedicated-token"


def test_internal_openai_gateway_falls_back_to_runtime_api_key():
    result = resolve_datahub_internal_openai_api_key(
        binding="openai",
        base_url="http://ai-service:8085/ai/internal/openai/v1",
        configured_api_key="provider-token",
        environ=_environment(),
    )

    assert result == "runtime-token"


def test_external_or_lookalike_endpoint_keeps_provider_key():
    external = resolve_datahub_internal_openai_api_key(
        binding="openai",
        base_url="https://provider.example/v1",
        configured_api_key="provider-token",
        environ=_environment(),
    )
    lookalike = resolve_datahub_internal_openai_api_key(
        binding="openai",
        base_url="http://ai-service.invalid:8085/ai/internal/openai/v1",
        configured_api_key="provider-token",
        environ=_environment(),
    )

    assert external == "provider-token"
    assert lookalike == "provider-token"


def test_non_openai_binding_keeps_configured_key():
    result = resolve_datahub_internal_openai_api_key(
        binding="azure_openai",
        base_url="http://ai-service:8085/ai/internal/openai/v1",
        configured_api_key="provider-token",
        environ=_environment(AI_SERVICE_INTERNAL_TOKEN="dedicated-token"),
    )

    assert result == "provider-token"
