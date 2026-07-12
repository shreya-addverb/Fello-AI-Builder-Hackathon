"""Google Gemini reasoning provider adapter."""

import json
import logging
import re
import asyncio
from time import perf_counter
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field, SecretStr

from backend.research.logging import (
    log_outbound_request,
    log_outbound_response,
    log_provider_completed,
    log_provider_failed,
    log_provider_started,
)
from backend.research.models import ReasonRequest, ReasonResponse, ResearchFailure
from backend.research.providers.support import (
    elapsed_milliseconds,
    exception_failure,
    missing_configuration_failure,
    qualified_exception_name,
)


class _GeminiPart(BaseModel):
    text: str | None = None


class _GeminiContent(BaseModel):
    parts: list[_GeminiPart]


class _GeminiCandidate(BaseModel):
    content: _GeminiContent


class _GeminiResponse(BaseModel):
    candidates: list[_GeminiCandidate]


class GeminiModelInfo(BaseModel):
    """Gemini model metadata returned by the Models API."""

    name: str
    display_name: str | None = None
    supported_generation_methods: list[str] = Field(default_factory=list)


class _GeminiAttempt(BaseModel):
    model: str
    status_code: int
    google_error: str


class GeminiProvider:
    """Perform reasoning and structured synthesis through Google Gemini."""

    name = "gemini"

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: SecretStr | None,
        generate_url: str | None,
        model: str | None,
        timeout_seconds: float | None,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._generate_url = generate_url
        self._model = model
        self._configured_model = model
        self._available_models: list[GeminiModelInfo] | None = None
        self._compatible_models: list[str] = []
        self._model_lock = asyncio.Lock()
        self._model_resolved = False
        self._model_generation = 0
        self._timeout_seconds = timeout_seconds

    async def reason(self, request: ReasonRequest) -> ReasonResponse:
        """Reason over supplied documents without invoking search tools."""
        operation = "reason"
        started_at = perf_counter()
        log_provider_started(self.name, operation)

        if not self._is_configured:
            return self._failure_response(
                missing_configuration_failure(self.name), started_at, operation
            )
        assert self._api_key is not None
        assert self._generate_url is not None
        assert self._model is not None
        assert self._timeout_seconds is not None

        resolution_failure = await self._resolve_model()
        if resolution_failure is not None:
            return self._failure_response(resolution_failure, started_at, operation)
        assert self._model is not None

        payload: dict[str, object] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": self._build_prompt(request)}],
                }
            ]
        }
        if request.output_mode == "json":
            payload["generationConfig"] = {
                "responseMimeType": "application/json",
                "responseJsonSchema": request.json_schema,
            }

        headers = {
            "x-goog-api-key": self._api_key.get_secret_value(),
            "Content-Type": "application/json",
        }
        diagnostic_logger = logging.getLogger("backend.research.gemini_diagnostic")
        try:
            attempts: list[_GeminiAttempt] = []
            response: httpx.Response | None = None
            url = ""
            models = list(self._compatible_models)
            model_generation = self._model_generation
            index = 0
            while index < len(models):
                model = models[index]
                url = self._generate_url.format(model=quote(model, safe=""))
                logging.getLogger("backend.research").info("Trying Gemini model: %s", model)
                log_outbound_request(self.name, "POST", url, self._timeout_seconds, headers, payload)
                diagnostic_logger.debug(
                    "Gemini request final_url=%s final_model=%s client_library=httpx "
                    "client_library_version=%s payload=%s", url, model, httpx.__version__,
                    json.dumps(payload, ensure_ascii=False, default=str),
                )
                response = await self._client.post(
                    url, headers=headers, json=payload, timeout=self._timeout_seconds
                )
                log_outbound_response(self.name, response.status_code, response.headers, response.text)
                google_error = self._google_error_message(response)
                diagnostic_logger.debug("Gemini raw response model=%s status=%d headers=%s body=%s", model, response.status_code, dict(response.headers), response.text)
                if response.is_success:
                    async with self._model_lock:
                        self._model = model
                        self._compatible_models = [model] + [item for item in models if item != model]
                        self._model_resolved = True
                    break
                attempts.append(_GeminiAttempt(model=model, status_code=response.status_code, google_error=google_error))
                if not self._is_failover_response(response):
                    break
                if index == 0:
                    resolution_failure = await self._resolve_model(
                        force=True,
                        stale_model=model,
                        stale_generation=model_generation,
                    )
                    if resolution_failure is not None:
                        return self._failure_response(resolution_failure, started_at, operation)
                    refreshed = tuple(self._compatible_models)
                    models = [model] + [item for item in refreshed if item != model]
                if index == len(models) - 1:
                    break
                logging.getLogger("backend.research").info(
                    "Gemini model %s unavailable (HTTP %d); trying fallback.",
                    model, response.status_code,
                )
                index += 1
            assert response is not None
            if not response.is_success:
                google_message = self._google_error_message(response)
                logging.getLogger("backend.research").error(
                    "Gemini request rejected requested_url=%s requested_model=%s "
                    "api_version=%s status=%d google_error=%r suggested_fix=%r",
                    url,
                    self._model,
                    self._api_version,
                    response.status_code,
                    google_message,
                    "Run check_gemini.py and select a listed model that supports generateContent.",
                )
                all_retryable = bool(attempts) and all(item.status_code in (404, 429) for item in attempts)
                report = {
                    "attempts": [item.model_dump() for item in attempts],
                    "final_reason": "All compatible Gemini models failed" if len(attempts) == len(self._compatible_models) else google_message,
                }
                report_json = json.dumps(report, ensure_ascii=False)
                return self._failure_response(
                    ResearchFailure(
                        provider=self.name,
                        code=("rate_limited" if response.status_code == 429 else "provider_error"),
                        message=report_json,
                        retryable=all_retryable,
                        status_code=response.status_code,
                        exception_type="httpx.HTTPStatusError",
                        response_body_excerpt=report_json,
                    ),
                    started_at,
                    operation,
                )
            provider_response = _GeminiResponse.model_validate(response.json())
            text = self._response_text(provider_response)
            if text is None:
                raise ValueError("Gemini response did not contain text.")
            structured_output = json.loads(text) if request.output_mode == "json" else None
            result = ReasonResponse(
                provider=self.name,
                model=self._model,
                text=text,
                structured_output=structured_output,
                duration_ms=elapsed_milliseconds(started_at),
            )
        except Exception as error:
            return self._failure_response(
                exception_failure(self.name, error),
                started_at,
                operation,
                error=error,
            )

        duration_ms = elapsed_milliseconds(started_at)
        log_provider_completed(self.name, operation, duration_ms)
        result.duration_ms = duration_ms
        return result

    async def list_available_models(self) -> list[GeminiModelInfo]:
        """List models visible to the configured key and cache their capabilities."""
        if self._available_models is not None:
            return self._available_models
        if self._api_key is None or self._generate_url is None or self._timeout_seconds is None:
            raise RuntimeError("Gemini configuration is incomplete.")
        models_url = self._generate_url.split("/models/", 1)[0] + "/models"
        response = await self._client.get(
            models_url,
            headers={"x-goog-api-key": self._api_key.get_secret_value()},
            timeout=self._timeout_seconds,
        )
        if not response.is_success:
            raise RuntimeError(
                f"Gemini Models API returned HTTP {response.status_code}: "
                f"{self._google_error_message(response)}"
            )
        payload = response.json()
        self._available_models = [
            GeminiModelInfo(
                name=item["name"],
                display_name=item.get("displayName"),
                supported_generation_methods=item.get("supportedGenerationMethods", []),
            )
            for item in payload.get("models", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        return self._available_models

    async def _resolve_model(
        self,
        *,
        force: bool = False,
        stale_model: str | None = None,
        stale_generation: int | None = None,
    ) -> ResearchFailure | None:
        """Resolve the configured name to a visible text model supporting generateContent."""
        async with self._model_lock:
            if self._model_resolved and not force:
                return None
            if force and (
                (stale_model is not None and self._model != stale_model)
                or (stale_generation is not None and self._model_generation != stale_generation)
            ):
                return None
            if force:
                self._available_models = None
            try:
                models = await self.list_available_models()
            except Exception as error:
                return ResearchFailure(
                    provider=self.name,
                    code="provider_error",
                    message=f"Unable to list Gemini models: {error}",
                    retryable=isinstance(error, httpx.RequestError),
                    exception_type=qualified_exception_name(error),
                    exception_message=str(error),
                )

            supported = [
                model
                for model in models
                if "generateContent" in model.supported_generation_methods
            ]
            configured = (self._configured_model or "").removeprefix("models/")
            candidates = [model for model in supported if self._is_general_text_model(model)]
            if not candidates:
                visible = ", ".join(model.name for model in supported) or "none"
                return ResearchFailure(
                provider=self.name,
                code="missing_configuration",
                    message=(
                        "No general-purpose Gemini model supporting generateContent is available. "
                        f"Models reporting generateContent: {visible}"
                    ),
                )

            configured_match = next(
                (model for model in candidates if model.name.removeprefix("models/") == configured),
                None,
            )
            ordered = sorted(candidates, key=self._model_preference)
            if configured_match is not None:
                ordered.remove(configured_match)
                ordered.insert(0, configured_match)
            self._compatible_models = [model.name.removeprefix("models/") for model in ordered]
            self._model = self._compatible_models[0]
            self._model_resolved = True
            self._model_generation += 1
            if configured_match is None:
                logging.getLogger("backend.research").warning(
                    "Configured Gemini model %r is unavailable; selected %r from the Models API.",
                    self._configured_model,
                    self._model,
                )
        return None

    @staticmethod
    def _is_general_text_model(model: GeminiModelInfo) -> bool:
        name = model.name.casefold()
        excluded = (
            "image", "embedding", "live", "tts", "audio", "robotics",
            "computer-use", "computer_use", "video", "predict",
        )
        return name.startswith("models/gemini-") and not any(part in name for part in excluded)

    @staticmethod
    def _model_preference(model: GeminiModelInfo) -> tuple[int, tuple[int, ...], str]:
        name = model.name.casefold()
        preview = any(marker in name for marker in ("preview", "experimental", "-exp"))
        if preview:
            tier = 4
        elif "flash-lite" in name or "flash_lite" in name:
            tier = 1
        elif "flash" in name:
            tier = 2
        else:
            tier = 3
        version_match = re.search(r"gemini-(\d+(?:\.\d+)*)", name)
        version = tuple(int(part) for part in version_match.group(1).split(".")) if version_match else ()
        return tier, tuple(-part for part in version), name

    @property
    def compatible_models(self) -> tuple[str, ...]:
        """Return eligible models in deterministic failover order after resolution."""
        return tuple(self._compatible_models)

    @property
    def selected_model(self) -> str | None:
        """Return the configured or resolved model name."""
        return self._model

    @property
    def api_version(self) -> str:
        """Return the API version encoded in the configured endpoint."""
        return self._api_version

    @property
    def _api_version(self) -> str:
        if self._generate_url is None:
            return "unknown"
        match = re.search(r"/(v\d+(?:beta\d*)?)/", self._generate_url)
        return match.group(1) if match else "unknown"

    @staticmethod
    def _google_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            message = error.get("message") if isinstance(error, dict) else None
            return str(message) if message else response.text
        except ValueError:
            return response.text

    @classmethod
    def _is_model_unavailable_response(cls, response: httpx.Response) -> bool:
        message = cls._google_error_message(response).casefold()
        return "model" in message and any(
            phrase in message
            for phrase in ("not found", "no longer available", "not supported")
        )

    @classmethod
    def _is_failover_response(cls, response: httpx.Response) -> bool:
        """Return whether this response permits immediate model failover."""
        return response.status_code == 429 or (
            response.status_code == 404 and cls._is_model_unavailable_response(response)
        )

    @property
    def _is_configured(self) -> bool:
        return bool(
            self._api_key
            and self._generate_url
            and "{model}" in self._generate_url
            and self._model
            and self._timeout_seconds
            and self._timeout_seconds > 0
        )

    @staticmethod
    def _build_prompt(request: ReasonRequest) -> str:
        sections = [request.instruction]
        for index, document in enumerate(request.documents, start=1):
            source = f"Source: {document.source}"
            if document.title is not None:
                source += f"\nTitle: {document.title}"
            if document.url is not None:
                source += f"\nURL: {document.url}"
            sections.append(f"Document {index}\n{source}\nContent:\n{document.content}")
        return "\n\n".join(sections)

    @staticmethod
    def _response_text(response: _GeminiResponse) -> str | None:
        if not response.candidates:
            return None
        parts = response.candidates[0].content.parts
        texts = [part.text for part in parts if part.text]
        return "".join(texts) or None

    def _failure_response(
        self,
        failure: ResearchFailure,
        started_at: float,
        operation: str,
        error: Exception | None = None,
    ) -> ReasonResponse:
        duration_ms = elapsed_milliseconds(started_at)
        log_provider_failed(
            self.name,
            operation,
            duration_ms,
            exception_type=(qualified_exception_name(error) if error else failure.exception_type),
            error_message=(str(error) if error else failure.message),
        )
        return ReasonResponse(
            provider=self.name,
            model=self._model,
            failure=failure,
            duration_ms=duration_ms,
        )
