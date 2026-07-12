"""Application configuration."""

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from pydantic import BaseModel, ConfigDict, SecretStr


_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_FILE)

# Accept the PowerShell assignment form used by the existing local environment file.
for _name, _value in dotenv_values(_ENV_FILE).items():
    if _name.startswith("$env:") and _value is not None:
        os.environ.setdefault(_name.removeprefix("$env:"), _value)


class Settings(BaseModel):
    """Static application metadata."""

    model_config = ConfigDict(frozen=True)

    app_name: str = "AI Account Intelligence API"
    app_version: str = "0.1.0"
    tavily_api_key: SecretStr | None = None
    tavily_search_url: str | None = None
    firecrawl_api_key: SecretStr | None = None
    firecrawl_scrape_url: str | None = None
    gemini_api_key: SecretStr | None = None
    gemini_generate_url: str | None = None
    gemini_model: str | None = None
    research_timeout_seconds: float | None = None
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")


def _secret_from_environment(name: str) -> SecretStr | None:
    """Read an optional secret without exposing it as a plain setting value."""
    value = os.getenv(name)
    return SecretStr(value) if value else None


def _float_from_environment(name: str) -> float | None:
    """Read an optional numeric environment setting."""
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


settings = Settings(
    tavily_api_key=_secret_from_environment("TAVILY_API_KEY"),
    tavily_search_url=os.getenv("TAVILY_SEARCH_URL"),
    firecrawl_api_key=_secret_from_environment("FIRECRAWL_API_KEY"),
    firecrawl_scrape_url=os.getenv("FIRECRAWL_SCRAPE_URL"),
    gemini_api_key=_secret_from_environment("GEMINI_API_KEY"),
    gemini_generate_url=os.getenv("GEMINI_GENERATE_URL"),
    gemini_model=os.getenv("GEMINI_MODEL"),
    research_timeout_seconds=_float_from_environment("RESEARCH_TIMEOUT_SECONDS"),
    cors_origins=tuple(
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
        if origin.strip()
    ),
)
