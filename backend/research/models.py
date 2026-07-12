"""Provider-neutral request and response models for research operations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, JsonValue, model_validator

FailureCode = Literal[
    "missing_configuration",
    "timeout",
    "rate_limited",
    "provider_error",
    "invalid_response",
]


class ResearchModel(BaseModel):
    """Strict base model for research-layer contracts."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ResearchFailure(ResearchModel):
    """Safe, structured details about a failed provider operation."""

    provider: str
    code: FailureCode
    message: str
    retryable: bool = False
    status_code: int | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    response_body_excerpt: str | None = None


class SearchRequest(ResearchModel):
    """Provider-neutral web search request."""

    query: str = Field(min_length=1)
    topic: Literal["general", "news"] = "general"
    max_results: int = Field(default=5, ge=1, le=20)
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)


class SearchResult(ResearchModel):
    """A normalized web search result."""

    title: str
    url: HttpUrl
    content: str
    score: float | None = Field(default=None, ge=0, le=1)
    published_date: str | None = None


class SearchResponse(ResearchModel):
    """Structured outcome of a web search operation."""

    provider: str
    results: list[SearchResult] = Field(default_factory=list)
    failure: ResearchFailure | None = None
    duration_ms: int = Field(ge=0)

    @property
    def succeeded(self) -> bool:
        """Return whether the provider completed without failure."""
        return self.failure is None


class CrawlRequest(ResearchModel):
    """Provider-neutral request for webpage content retrieval."""

    url: HttpUrl
    only_main_content: bool = True


class CrawlMetadata(ResearchModel):
    """Normalized metadata returned with crawled page content."""

    title: str | None = None
    description: str | None = None
    source_url: HttpUrl | None = None
    status_code: int | None = None


class CrawlResponse(ResearchModel):
    """Structured outcome of a webpage crawl operation."""

    provider: str
    markdown: str | None = None
    metadata: CrawlMetadata | None = None
    failure: ResearchFailure | None = None
    duration_ms: int = Field(ge=0)

    @property
    def succeeded(self) -> bool:
        """Return whether the provider completed without failure."""
        return self.failure is None


class ResearchDocument(ResearchModel):
    """A source document supplied to the reasoning provider."""

    content: str = Field(min_length=1)
    source: str
    title: str | None = None
    url: HttpUrl | None = None


class ReasonRequest(ResearchModel):
    """Provider-neutral reasoning and synthesis request."""

    instruction: str = Field(min_length=1)
    documents: list[ResearchDocument] = Field(default_factory=list)
    output_mode: Literal["text", "json"] = "text"
    json_schema: JsonValue | None = None

    @model_validator(mode="after")
    def validate_json_output(self) -> "ReasonRequest":
        """Require a schema only when structured JSON output is requested."""
        if self.output_mode == "json" and self.json_schema is None:
            raise ValueError("json_schema is required when output_mode is json.")
        if self.output_mode == "text" and self.json_schema is not None:
            raise ValueError("json_schema is only valid when output_mode is json.")
        return self


class ReasonResponse(ResearchModel):
    """Structured outcome of a reasoning operation."""

    provider: str
    model: str | None = None
    text: str | None = None
    structured_output: JsonValue | None = None
    failure: ResearchFailure | None = None
    duration_ms: int = Field(ge=0)

    @property
    def succeeded(self) -> bool:
        """Return whether the provider completed without failure."""
        return self.failure is None
