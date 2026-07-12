"""Firecrawl webpage content provider adapter."""

from time import perf_counter

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr

from backend.research.logging import (
    log_outbound_request,
    log_outbound_response,
    log_provider_completed,
    log_provider_failed,
    log_provider_started,
)
from backend.research.models import (
    CrawlMetadata,
    CrawlRequest,
    CrawlResponse,
    ResearchFailure,
)
from backend.research.providers.support import (
    elapsed_milliseconds,
    exception_failure,
    http_status_failure,
    missing_configuration_failure,
    qualified_exception_name,
)


class _FirecrawlMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    title: str | None = None
    description: str | None = None
    source_url: HttpUrl | None = Field(default=None, alias="sourceURL")
    status_code: int | None = Field(default=None, alias="statusCode")


class _FirecrawlData(BaseModel):
    markdown: str | None = None
    metadata: _FirecrawlMetadata | None = None


class _FirecrawlResponse(BaseModel):
    success: bool
    data: _FirecrawlData | None = None


class FirecrawlProvider:
    """Retrieve webpage Markdown and metadata through Firecrawl."""

    name = "firecrawl"

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: SecretStr | None,
        scrape_url: str | None,
        timeout_seconds: float | None,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._scrape_url = scrape_url
        self._timeout_seconds = timeout_seconds

    async def crawl(self, request: CrawlRequest) -> CrawlResponse:
        """Retrieve the main Markdown content for a single webpage."""
        operation = "crawl"
        started_at = perf_counter()
        log_provider_started(self.name, operation)

        if not self._is_configured:
            return self._failure_response(
                missing_configuration_failure(self.name), started_at, operation
            )
        assert self._api_key is not None
        assert self._scrape_url is not None
        assert self._timeout_seconds is not None

        payload = {
            "url": str(request.url),
            "formats": ["markdown"],
            "onlyMainContent": request.only_main_content,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        log_outbound_request(
            self.name, "POST", self._scrape_url, self._timeout_seconds, headers, payload
        )
        try:
            response = await self._client.post(
                self._scrape_url,
                headers=headers,
                json=payload,
                timeout=self._timeout_seconds,
            )
            log_outbound_response(
                self.name, response.status_code, response.headers, response.text
            )
            if not response.is_success:
                return self._failure_response(
                    http_status_failure(self.name, response.status_code, response.text),
                    started_at,
                    operation,
                )
            provider_response = _FirecrawlResponse.model_validate(response.json())
            if not provider_response.success or provider_response.data is None:
                return self._failure_response(
                    ResearchFailure(
                        provider=self.name,
                        code="provider_error",
                        message="Provider did not return crawled content.",
                    ),
                    started_at,
                    operation,
                )
            if provider_response.data.markdown is None:
                raise ValueError("Firecrawl response did not contain Markdown.")
            data = provider_response.data
            metadata = data.metadata
            normalized_metadata = (
                CrawlMetadata(
                    title=metadata.title,
                    description=metadata.description,
                    source_url=metadata.source_url,
                    status_code=metadata.status_code,
                )
                if metadata is not None
                else None
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
        return CrawlResponse(
            provider=self.name,
            markdown=data.markdown,
            metadata=normalized_metadata,
            duration_ms=duration_ms,
        )

    @property
    def _is_configured(self) -> bool:
        return bool(
            self._api_key
            and self._scrape_url
            and self._timeout_seconds
            and self._timeout_seconds > 0
        )

    def _failure_response(
        self,
        failure: ResearchFailure,
        started_at: float,
        operation: str,
        error: Exception | None = None,
    ) -> CrawlResponse:
        duration_ms = elapsed_milliseconds(started_at)
        log_provider_failed(
            self.name,
            operation,
            duration_ms,
            exception_type=(qualified_exception_name(error) if error else failure.exception_type),
            error_message=(str(error) if error else failure.message),
        )
        return CrawlResponse(
            provider=self.name,
            failure=failure,
            duration_ms=duration_ms,
        )
