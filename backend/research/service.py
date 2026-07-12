"""Provider-neutral facade for shared research capabilities."""

import ipaddress
from typing import Any

import httpx

from backend.research.models import (
    CrawlRequest,
    CrawlResponse,
    ReasonRequest,
    ReasonResponse,
    SearchRequest,
    SearchResponse,
)
from backend.research.protocols import CrawlProvider, ReasoningProvider, SearchProvider


class ResearchService:
    """Expose high-level research operations while hiding provider details."""

    def __init__(
        self,
        search_provider: SearchProvider,
        crawl_provider: CrawlProvider,
        reasoning_provider: ReasoningProvider,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._search_provider = search_provider
        self._crawl_provider = crawl_provider
        self._reasoning_provider = reasoning_provider
        self._client = client
        self._cache: dict[tuple[str, str], Any] = {}
        self._trace: list[dict[str, Any]] = []

    @property
    def trace_count(self) -> int:
        return len(self._trace)

    def trace_since(self, index: int) -> list[dict[str, Any]]:
        return self._trace[index:]

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Retrieve normalized search results."""
        key = ("search", request.model_dump_json())
        if key in self._cache:
            self._trace.append({"provider": "cache", "model": None, "retry_count": 0, "fallback": "reused cached search", "error": None})
            return self._cache[key]
        response = await self._search_provider.search(request)
        self._cache[key] = response
        self._record(response)
        return response

    async def crawl(self, request: CrawlRequest) -> CrawlResponse:
        """Retrieve normalized webpage content."""
        key = ("crawl", request.model_dump_json())
        if key in self._cache:
            self._trace.append({"provider": "cache", "model": None, "retry_count": 0, "fallback": "reused cached crawl", "error": None})
            return self._cache[key]
        response = await self._crawl_provider.crawl(request)
        self._cache[key] = response
        self._record(response)
        return response

    async def reason(self, request: ReasonRequest) -> ReasonResponse:
        """Reason over caller-supplied research documents."""
        response = await self._reasoning_provider.reason(request)
        self._record(response)
        return response

    def _record(self, response: Any) -> None:
        failure = getattr(response, "failure", None)
        self._trace.append({
            "provider": getattr(response, "provider", None),
            "model": getattr(response, "model", None),
            "retry_count": 0,
            "fallback": None,
            "error": f"{failure.code}: {failure.message}" if failure else None,
        })

    async def lookup_ip_organization(self, ip: str) -> tuple[str | None, str | None]:
        """Resolve public visitor IP ownership and location through ipapi.co."""
        address = ipaddress.ip_address(ip)
        if address.is_private or address.is_loopback or address.is_reserved or self._client is None:
            return None, None
        try:
            response = await self._client.get(
                f"https://ipapi.co/{address}/json/",
                timeout=10.0,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                return None, None
            organization = payload.get("org")
            location = ", ".join(
                str(value) for value in (payload.get("city"), payload.get("region"), payload.get("country_name")) if value
            ) or None
            return (str(organization).strip() if organization else None), location
        except (httpx.HTTPError, ValueError, TypeError):
            return None, None
