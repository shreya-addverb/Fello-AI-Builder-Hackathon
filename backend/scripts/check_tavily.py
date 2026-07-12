"""Perform one direct Tavily API call without the pipeline or ResearchService."""

import asyncio

from backend.config import settings
from backend.research.http_client import create_async_http_client
from backend.research.models import SearchRequest
from backend.research.providers.tavily import TavilyProvider
from backend.scripts.connectivity import report_network


async def main() -> None:
    """Run network diagnostics and exactly one Tavily search request."""
    if settings.tavily_search_url is None:
        raise RuntimeError("TAVILY_SEARCH_URL is not configured.")
    report_network(settings.tavily_search_url)
    async with create_async_http_client() as client:
        result = await TavilyProvider(
            client,
            settings.tavily_api_key,
            settings.tavily_search_url,
            settings.research_timeout_seconds,
        ).search(SearchRequest(query="Amazon company official website", max_results=1))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
