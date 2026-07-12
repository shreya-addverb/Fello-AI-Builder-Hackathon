"""Default external provider adapters for the research layer."""

from backend.research.providers.firecrawl import FirecrawlProvider
from backend.research.providers.gemini import GeminiProvider
from backend.research.providers.tavily import TavilyProvider

__all__ = ["FirecrawlProvider", "GeminiProvider", "TavilyProvider"]
