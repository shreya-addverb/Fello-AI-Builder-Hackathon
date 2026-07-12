"""Explicit composition for the default research provider implementations."""

import httpx

from backend.config import Settings, settings
from backend.research.providers import FirecrawlProvider, GeminiProvider, TavilyProvider
from backend.research.service import ResearchService


def create_research_service(
    client: httpx.AsyncClient,
    application_settings: Settings = settings,
) -> ResearchService:
    """Compose the research facade from environment-configured providers."""
    return ResearchService(
        search_provider=TavilyProvider(
            client=client,
            api_key=application_settings.tavily_api_key,
            search_url=application_settings.tavily_search_url,
            timeout_seconds=application_settings.research_timeout_seconds,
        ),
        crawl_provider=FirecrawlProvider(
            client=client,
            api_key=application_settings.firecrawl_api_key,
            scrape_url=application_settings.firecrawl_scrape_url,
            timeout_seconds=application_settings.research_timeout_seconds,
        ),
        reasoning_provider=GeminiProvider(
            client=client,
            api_key=application_settings.gemini_api_key,
            generate_url=application_settings.gemini_generate_url,
            model=application_settings.gemini_model,
            timeout_seconds=application_settings.research_timeout_seconds,
        ),
        client=client,
    )
