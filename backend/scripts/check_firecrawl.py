"""Perform one direct Firecrawl API call without the pipeline or ResearchService."""

import asyncio

from backend.config import settings
from backend.research.http_client import create_async_http_client
from backend.research.models import CrawlRequest
from backend.research.providers.firecrawl import FirecrawlProvider
from backend.scripts.connectivity import report_network


async def main() -> None:
    """Run network diagnostics and exactly one Firecrawl scrape request."""
    if settings.firecrawl_scrape_url is None:
        raise RuntimeError("FIRECRAWL_SCRAPE_URL is not configured.")
    report_network(settings.firecrawl_scrape_url)
    async with create_async_http_client() as client:
        result = await FirecrawlProvider(
            client,
            settings.firecrawl_api_key,
            settings.firecrawl_scrape_url,
            settings.research_timeout_seconds,
        ).crawl(CrawlRequest(url="https://example.com"))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
