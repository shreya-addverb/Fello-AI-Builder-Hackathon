"""Replaceable provider interfaces used by the research service."""

from typing import Protocol

from backend.research.models import (
    CrawlRequest,
    CrawlResponse,
    ReasonRequest,
    ReasonResponse,
    SearchRequest,
    SearchResponse,
)


class SearchProvider(Protocol):
    """Provider capable of retrieving web search results."""

    async def search(self, request: SearchRequest) -> SearchResponse: ...


class CrawlProvider(Protocol):
    """Provider capable of retrieving webpage content."""

    async def crawl(self, request: CrawlRequest) -> CrawlResponse: ...


class ReasoningProvider(Protocol):
    """Provider capable of reasoning over supplied research documents."""

    async def reason(self, request: ReasonRequest) -> ReasonResponse: ...
