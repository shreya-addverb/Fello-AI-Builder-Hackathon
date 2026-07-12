"""Strongly typed shared state for the account intelligence pipeline."""

from datetime import date, datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    IPvAnyAddress,
    JsonValue,
    HttpUrl,
    model_validator,
)

PipelineStatus = Literal["pending", "running", "completed", "failed"]
StageStatus = Literal["completed", "no_data", "failed", "skipped"]


class ContextModel(BaseModel):
    """Base model for validated, assignment-safe pipeline state."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class AnalysisInput(ContextModel):
    """Normalized request data supplied to the pipeline."""

    visitor_id: str | None = Field(default=None, min_length=1)
    ip_address: IPvAnyAddress | None = None
    company_name: str | None = Field(default=None, min_length=1)
    domain: str | None = Field(default=None, min_length=1)
    pages_visited: list[str] = Field(default_factory=list)
    visit_duration: int | None = Field(default=None, ge=0)
    visits_this_week: int | None = Field(default=None, ge=0)
    referral_source: str | None = None
    device_type: str | None = None
    visitor_location: str | None = None
    visit_timestamp: datetime | None = None


class CompanyIdentification(ContextModel):
    """Output owned by the company identification stage."""

    identified_company: str | None = None
    identified_domain: str | None = None
    identification_confidence: float | None = Field(default=None, ge=0, le=1)
    reasoning: str | None = None
    evidence: list["EvidenceReference"] = Field(default_factory=list)


class EvidenceReference(ContextModel):
    """Public provenance attached to a claim or group of claims."""

    source_url: HttpUrl
    provider: str
    evidence: str | None = None


class CompanyEnrichment(ContextModel):
    """Output owned by the company enrichment stage."""

    canonical_company_name: str | None = None
    website: HttpUrl | None = None
    industry: str | None = None
    business_category: str | None = None
    company_size: str | None = None
    employee_count: int | None = Field(default=None, ge=0)
    headquarters: str | None = None
    founded_year: int | None = Field(default=None, ge=0)
    business_description: str | None = None
    ownership_type: Literal["public", "private", "government", "nonprofit", "unknown"] | None = None
    revenue: str | None = None
    stock_ticker: str | None = None
    geographic_footprint: list[str] = Field(default_factory=list)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    field_evidence: dict[str, list[EvidenceReference]] = Field(default_factory=dict)
    enrichment_confidence: float | None = Field(default=None, ge=0, le=1)


class DetectedTechnology(ContextModel):
    """A technology supported by evidence with detection certainty."""

    name: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    source: str | None = None
    evidence: list[EvidenceReference] = Field(default_factory=list)


class TechnologyStack(ContextModel):
    """Evidence-backed technologies grouped by functional category."""

    crm: list[DetectedTechnology] = Field(default_factory=list)
    marketing: list[DetectedTechnology] = Field(default_factory=list)
    analytics: list[DetectedTechnology] = Field(default_factory=list)
    cms: list[DetectedTechnology] = Field(default_factory=list)
    frontend: list[DetectedTechnology] = Field(default_factory=list)
    backend: list[DetectedTechnology] = Field(default_factory=list)
    hosting: list[DetectedTechnology] = Field(default_factory=list)
    cloud: list[DetectedTechnology] = Field(default_factory=list)
    security: list[DetectedTechnology] = Field(default_factory=list)
    databases: list[DetectedTechnology] = Field(default_factory=list)
    ai_platforms: list[DetectedTechnology] = Field(default_factory=list)
    developer_tools: list[DetectedTechnology] = Field(default_factory=list)
    customer_support: list[DetectedTechnology] = Field(default_factory=list)
    other: list[DetectedTechnology] = Field(default_factory=list)
    detection_confidence: float | None = Field(default=None, ge=0, le=1)


class Leader(ContextModel):
    """A company leader discovered during research."""

    full_name: str = Field(min_length=1)
    job_title: str = Field(min_length=1)
    department: str | None = None
    organization: str = Field(min_length=1)
    linkedin_url: HttpUrl | None = None
    source_url: HttpUrl
    confidence: float | None = Field(default=None, ge=0, le=1)
    provider: str | None = None
    evidence: str | None = None


class Leadership(ContextModel):
    """Output owned by the leadership discovery stage."""

    leaders: list[Leader] = Field(default_factory=list)
    discovery_confidence: float | None = Field(default=None, ge=0, le=1)


class BusinessSignal(ContextModel):
    """A relevant, sourced company event or signal."""

    signal_type: Literal[
        "hiring",
        "funding",
        "expansion",
        "product",
        "partnership",
        "recognition",
        "growth",
        "other",
        "acquisition",
        "ai_initiative",
        "cloud_migration",
    ]
    title: str = Field(min_length=1)
    description: str
    event_date: date | None = None
    source_url: HttpUrl
    confidence: float | None = Field(default=None, ge=0, le=1)
    importance: Literal["low", "medium", "high", "critical"] | None = None
    provider: str | None = None
    evidence: str | None = None


class BusinessSignals(ContextModel):
    """Output owned by the business signals stage."""

    signals: list[BusinessSignal] = Field(default_factory=list)
    overall_confidence: float | None = Field(default=None, ge=0, le=1)


class InferredPersona(ContextModel):
    """One evidence-supported visitor persona assessment."""

    likely_persona: str = Field(min_length=1)
    department: str | None = None
    seniority: str | None = None
    reasoning: str = Field(min_length=1)
    supporting_signals: list[str] = Field(min_length=2)
    confidence: float = Field(ge=0, le=1)


class Persona(ContextModel):
    """Output owned by the persona inference stage."""

    personas: list[InferredPersona] = Field(default_factory=list)
    overall_persona_confidence: float | None = Field(default=None, ge=0, le=1)


class Intent(ContextModel):
    """Output owned by the intent scoring stage."""

    intent_score: float = Field(default=0, ge=0, le=10)
    intent_stage: Literal[
        "Awareness",
        "Research",
        "Consideration",
        "Evaluation",
        "Decision",
        "Unknown",
    ] = "Unknown"
    confidence: float = Field(default=0, ge=0, le=1)
    supporting_signals: list[str] = Field(default_factory=list)
    reasoning_summary: str | None = None

    @model_validator(mode="after")
    def validate_intent_state(self) -> "Intent":
        """Keep unknown and evidence-backed intent states internally consistent."""
        if self.intent_stage == "Unknown":
            if self.intent_score != 0 or self.confidence != 0:
                raise ValueError("Unknown intent must have zero score and confidence.")
        elif len(self.supporting_signals) < 2:
            raise ValueError("Known intent requires at least two supporting signals.")
        return self


class AISummary(ContextModel):
    """Output owned by the summary generation stage."""

    executive_summary: str | None = None
    company_overview: str | None = None
    technology_overview: str | None = None
    leadership_overview: str | None = None
    business_activity_overview: str | None = None
    visitor_assessment: str | None = None
    buying_intent_assessment: str | None = None
    key_opportunities: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0, ge=0, le=1)


class RecommendedContact(ContextModel):
    """An existing discovered leader recommended for outreach."""

    full_name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    reason_for_contact: str = Field(min_length=1)


class SalesRecommendations(ContextModel):
    """Structured sales recommendations for the analyzed account."""

    account_priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] | None = None
    recommended_contacts: list[RecommendedContact] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    outreach_strategy: str | None = None
    messaging_points: list[str] = Field(default_factory=list)
    recommended_products: list[str] = Field(default_factory=list)
    urgency: Literal["LOW", "NORMAL", "HIGH", "IMMEDIATE"] | None = None
    reasoning_summary: str | None = None
    confidence: float = Field(default=0, ge=0, le=1)


class ConfidenceScores(ContextModel):
    """Stage confidence values not already owned by another section."""


class RawResearchArtifact(ContextModel):
    """An internal intermediate output retained for later processing."""

    stage: str
    content: JsonValue


class RawResearch(ContextModel):
    """Internal research artifacts not intended as a public report contract."""

    artifacts: list[RawResearchArtifact] = Field(default_factory=list)


class PipelineMetadata(ContextModel):
    """Execution state maintained by the pipeline itself."""

    status: PipelineStatus = "pending"
    completed_steps: list[str] = Field(default_factory=list)
    stage_results: list["StageResult"] = Field(default_factory=list)


class StageResult(ContextModel):
    """Observable outcome for one pipeline stage."""

    stage: str
    status: StageStatus
    duration_ms: int = Field(ge=0)
    message: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())
    ended_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())
    providers_used: list[str] = Field(default_factory=list)
    models_used: list[str] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)
    fallback_events: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class AnalysisContext(ContextModel):
    """Single source of truth progressively enriched by pipeline services."""

    request_type: Literal["company", "visitor"]
    input: AnalysisInput
    company_identification: CompanyIdentification = Field(
        default_factory=CompanyIdentification
    )
    company_enrichment: CompanyEnrichment = Field(default_factory=CompanyEnrichment)
    technology_stack: TechnologyStack = Field(default_factory=TechnologyStack)
    leadership: Leadership = Field(default_factory=Leadership)
    business_signals: BusinessSignals = Field(default_factory=BusinessSignals)
    persona: Persona = Field(default_factory=Persona)
    intent: Intent = Field(default_factory=Intent)
    ai_summary: AISummary = Field(default_factory=AISummary)
    sales_recommendations: SalesRecommendations = Field(
        default_factory=SalesRecommendations
    )
    confidence: ConfidenceScores = Field(default_factory=ConfidenceScores)
    raw_research: RawResearch | None = Field(default=None, exclude=True)
    pipeline_metadata: PipelineMetadata = Field(default_factory=PipelineMetadata)

    @model_validator(mode="after")
    def validate_request_input(self) -> "AnalysisContext":
        """Reject input combinations that contradict the requested analysis type."""
        if self.request_type == "company":
            if self.input.company_name is None and self.input.domain is None:
                raise ValueError("Company analysis requires a company name or domain.")

            visitor_fields = (
                self.input.visitor_id,
                self.input.ip_address,
                self.input.visit_duration,
                self.input.visits_this_week,
                self.input.referral_source,
                self.input.device_type,
                self.input.visitor_location,
                self.input.visit_timestamp,
            )
            if any(value is not None for value in visitor_fields) or self.input.pages_visited:
                raise ValueError("Company analysis cannot contain visitor signals.")

        if self.request_type == "visitor":
            required_visitor_fields = (
                self.input.visitor_id,
                self.input.ip_address,
                self.input.visit_duration,
                self.input.visits_this_week,
            )
            if any(value is None for value in required_visitor_fields):
                raise ValueError(
                    "Visitor analysis requires visitor_id, ip_address, "
                    "visit_duration, and visits_this_week."
                )
            if self.input.company_name is not None:
                raise ValueError("Visitor analysis cannot contain a company name.")

        return self
