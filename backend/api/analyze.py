"""Analysis API endpoints."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends

from backend.models.context import AnalysisContext, AnalysisInput
from backend.models.schemas import (
    CompanyRequest,
    PipelineResponse,
    VisitorRequest,
)
from backend.research.dependencies import create_research_service
from backend.research.http_client import create_async_http_client
from backend.research.service import ResearchService
from backend.services.business_signals import BusinessSignals
from backend.services.company_enrichment import CompanyEnrichment
from backend.services.company_identifier import CompanyIdentifier
from backend.services.intent_scoring import IntentScoring
from backend.services.leadership_discovery import LeadershipDiscovery
from backend.services.persona_inference import PersonaInference
from backend.services.sales_recommendations import SalesRecommendationGeneration
from backend.services.pipeline import AccountIntelligencePipeline
from backend.services.summary_generation import SummaryGeneration
from backend.services.technology_detection import TechnologyDetection

router = APIRouter(prefix="/analyze", tags=["Analysis"])


async def get_research_service() -> AsyncIterator[ResearchService]:
    """Provide a request-scoped research service and managed HTTP client."""
    async with create_async_http_client() as client:
        yield create_research_service(client)


ResearchDependency = Annotated[ResearchService, Depends(get_research_service)]


def get_pipeline(research_service: ResearchDependency) -> AccountIntelligencePipeline:
    """Compose and provide the account intelligence pipeline."""
    return AccountIntelligencePipeline(
        company_identifier=CompanyIdentifier(research_service=research_service),
        company_enrichment=CompanyEnrichment(research_service=research_service),
        technology_detection=TechnologyDetection(research_service=research_service),
        leadership_discovery=LeadershipDiscovery(research_service=research_service),
        business_signals=BusinessSignals(research_service=research_service),
        persona_inference=PersonaInference(research_service=research_service),
        intent_scoring=IntentScoring(research_service=research_service),
        summary_generation=SummaryGeneration(
            research_service=research_service,
            sales_recommendation_service=SalesRecommendationGeneration(
                research_service=research_service
            ),
        ),
    )


PipelineDependency = Annotated[AccountIntelligencePipeline, Depends(get_pipeline)]


def build_pipeline_response(context: AnalysisContext) -> PipelineResponse:
    """Translate authoritative application state into the API response contract."""
    return PipelineResponse(
        pipeline_status=context.pipeline_metadata.status,
        completed_steps=context.pipeline_metadata.completed_steps,
        current_context=context,
    )


@router.post("/company", response_model=PipelineResponse)
async def analyze_company(
    request: CompanyRequest,
    pipeline: PipelineDependency,
) -> PipelineResponse:
    """Run the pipeline architecture for a company request."""
    context = AnalysisContext(
        request_type="company",
        input=AnalysisInput(
            company_name=request.company_name,
            domain=request.domain,
        ),
    )
    context = await pipeline.execute(context)
    return build_pipeline_response(context)


@router.post("/visitor", response_model=PipelineResponse)
async def analyze_visitor(
    request: VisitorRequest,
    pipeline: PipelineDependency,
) -> PipelineResponse:
    """Run the pipeline architecture for a visitor request."""
    context = AnalysisContext(
        request_type="visitor",
        input=AnalysisInput(
            visitor_id=request.visitor_id,
            ip_address=request.ip,
            domain=request.domain,
            pages_visited=request.pages_visited,
            visit_duration=request.time_on_site_seconds,
            visits_this_week=request.visits_this_week,
            referral_source=request.referral_source,
            device_type=request.device_type,
            visitor_location=request.visitor_location,
            visit_timestamp=request.visit_timestamp,
        ),
    )
    context = await pipeline.execute(context)
    return build_pipeline_response(context)
