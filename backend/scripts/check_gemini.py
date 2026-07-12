"""Perform one direct Gemini API call without the pipeline or ResearchService."""

import asyncio
from time import perf_counter
from urllib.parse import quote

import httpx

from backend.config import settings
from backend.research.http_client import create_async_http_client
from backend.research.providers.gemini import GeminiProvider
from backend.scripts.connectivity import report_network


async def main() -> None:
    """Probe every compatible Gemini generation model."""
    if settings.gemini_generate_url is None:
        raise RuntimeError("GEMINI_GENERATE_URL is not configured.")
    report_network(settings.gemini_generate_url.replace("{model}", settings.gemini_model or "model"))
    async with create_async_http_client() as client:
        provider = GeminiProvider(
            client,
            settings.gemini_api_key,
            settings.gemini_generate_url,
            settings.gemini_model,
            settings.research_timeout_seconds,
        )
        try:
            models = await provider.list_available_models()
        except Exception as error:
            print(f"Unable to list Gemini models: {type(error).__name__}: {error}")
            return
        print(f"API version: {provider.api_version}")
        print("Available models:")
        for model in models:
            print(
                f"- {model.name} | displayName={model.display_name!r} | "
                f"supportedGenerationMethods={model.supported_generation_methods}"
            )
        resolution_failure = await provider._resolve_model()
        if resolution_failure is not None:
            print(resolution_failure.model_dump_json(indent=2))
            return

        best_model: str | None = None
        assert settings.gemini_api_key is not None
        for model in provider.compatible_models:
            started_at = perf_counter()
            try:
                response = await client.post(
                    settings.gemini_generate_url.format(model=quote(model, safe="")),
                    headers={
                        "x-goog-api-key": settings.gemini_api_key.get_secret_value(),
                        "Content-Type": "application/json",
                    },
                    json={"contents": [{"role": "user", "parts": [{"text": "Reply with the single word OK."}]}]},
                    timeout=settings.research_timeout_seconds,
                )
                status: int | str = response.status_code
                passed = response.is_success
            except httpx.RequestError as error:
                status = type(error).__name__
                passed = False
            latency_ms = round((perf_counter() - started_at) * 1000)
            print(f"{model} | {'PASS' if passed else 'FAIL'} | STATUS {status} | LATENCY {latency_ms}ms")
            if passed and best_model is None:
                best_model = model

        print(f"Best working model: {best_model or 'none'}")


if __name__ == "__main__":
    asyncio.run(main())
