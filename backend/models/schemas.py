"""Request and response schemas for the Account Intelligence API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, IPvAnyAddress, model_validator

from backend.models.context import AnalysisContext, PipelineStatus


class APIModel(BaseModel):
    """Base schema with strict handling of unexpected fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CompanyRequest(APIModel):
    """Request to initialize intelligence analysis for a company."""

    company_name: str | None = Field(default=None, min_length=1)
    domain: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_company_signal(self) -> "CompanyRequest":
        """Require at least one company identity signal."""
        if self.company_name is None and self.domain is None:
            raise ValueError("company_name or domain is required.")
        return self


class VisitorRequest(APIModel):
    """Website visitor signals used to initialize an analysis request."""

    visitor_id: str = Field(min_length=1)
    ip: IPvAnyAddress
    pages_visited: list[str]
    time_on_site_seconds: int = Field(ge=0)
    visits_this_week: int = Field(ge=0)
    domain: str | None = Field(default=None, min_length=1)
    referral_source: str | None = None
    device_type: str | None = None
    visitor_location: str | None = None
    visit_timestamp: datetime | None = None


class PipelineResponse(APIModel):
    """Debug response describing a completed pipeline execution."""

    pipeline_status: PipelineStatus
    completed_steps: list[str]
    current_context: AnalysisContext 