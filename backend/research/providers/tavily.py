"""Tavily web search provider adapter."""

from time import perf_counter

import httpx
from pydantic import BaseModel, HttpUrl, SecretStr

from backend.research.logging import (
    log_outbound_request,
    log_outbound_response,
    log_provider_completed,
    log_provider_failed,
    log_provider_started,
)
from backend.research.models import (
    ResearchFailure,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from backend.research.providers.support import (
    elapsed_milliseconds,
    exception_failure,
    http_status_failure,
    missing_configuration_failure,
    qualified_exception_name,
)


class _TavilySearchResult(BaseModel):
    title: str
    url: HttpUrl
    content: str
    score: float | None = None
    published_date: str | None = None


class _TavilySearchResponse(BaseModel):
    results: list[_TavilySearchResult]


class TavilyProvider:
    """Retrieve normalized web and news search results from Tavily."""

    name = "tavily"

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: SecretStr | None,
        search_url: str | None,
        timeout_seconds: float | None,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._search_url = search_url
        self._timeout_seconds = timeout_seconds

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute a Tavily search and normalize its result set."""
        operation = "search"
        started_at = perf_counter()
        log_provider_started(self.name, operation)

        if not self._is_configured:
            return self._failure_response(
                missing_configuration_failure(self.name), started_at, operation
            )
        assert self._api_key is not None
        assert self._search_url is not None
        assert self._timeout_seconds is not None

        payload: dict[str, object] = {
            "query": request.query,
            "topic": request.topic,
            "max_results": request.max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        if request.include_domains:
            payload["include_domains"] = request.include_domains
        if request.exclude_domains:
            payload["exclude_domains"] = request.exclude_domains

        headers = {
            "Authorization": f"Bearer {self._api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        log_outbound_request(
            self.name, "POST", self._search_url, self._timeout_seconds, headers, payload
        )
        try:
            response = await self._client.post(
                self._search_url,
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
            provider_response = _TavilySearchResponse.model_validate(response.json())
            results = [
                SearchResult(
                    title=result.title,
                    url=result.url,
                    content=result.content,
                    score=result.score,
                    published_date=result.published_date,
                )
                for result in provider_response.results
            ]
        except Exception as error:
            return self._failure_response(
                exception_failure(self.name, error),
                started_at,
                operation,
                error=error,
            )

        duration_ms = elapsed_milliseconds(started_at)
        log_provider_completed(self.name, operation, duration_ms)
        return SearchResponse(
            provider=self.name,
            results=results,
            duration_ms=duration_ms,
        )

    @property
    def _is_configured(self) -> bool:
        return bool(
            self._api_key
            and self._search_url
            and self._timeout_seconds
            and self._timeout_seconds > 0
        )

    def _failure_response(
        self,
        failure: ResearchFailure,
        started_at: float,
        operation: str,
        error: Exception | None = None,
    ) -> SearchResponse:
        duration_ms = elapsed_milliseconds(started_at)
        log_provider_failed(
            self.name,
            operation,
            duration_ms,
            exception_type=(qualified_exception_name(error) if error else failure.exception_type),
            error_message=(str(error) if error else failure.message),
        )
        return SearchResponse(
            provider=self.name,
            failure=failure,
            duration_ms=duration_ms,
        )
