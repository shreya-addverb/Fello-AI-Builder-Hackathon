"""Regression coverage for separating visitors from researched companies."""

from copy import deepcopy

import pytest

from backend.models.context import (
    AnalysisContext,
    AnalysisInput,
    BusinessSignal,
    DetectedTechnology,
    Leader,
)
from backend.research.models import ReasonResponse
from backend.services.persona_inference import PersonaInference


class _CapturingResearch:
    def __init__(self, output: dict | None = None) -> None:
        self.requests = []
        self.output = output or {
            "personas": [
                {
                    "likely_persona": "Technical Evaluator",
                    "department": "Technology",
                    "seniority": None,
                    "reasoning": "Product research and documentation activity indicate technical evaluation.",
                    "supporting_signal_keys": [
                        "behavior.page.0",
                        "behavior.page.1",
                    ],
                    "confidence": 0.8,
                }
            ],
            "overall_confidence": 0.8,
        }

    async def reason(self, request):
        self.requests.append(request)
        return ReasonResponse(
            provider="test",
            structured_output=self.output,
            duration_ms=0,
        )


def _visitor_context() -> AnalysisContext:
    context = AnalysisContext(
        request_type="visitor",
        input=AnalysisInput(
            visitor_id="anonymous-1",
            ip_address="8.8.8.8",
            pages_visited=["/pricing", "/docs/api", "/case-studies"],
            visit_duration=420,
            visits_this_week=4,
            referral_source="search",
        ),
    )
    context.company_enrichment.industry = "Healthcare"
    context.company_enrichment.company_size = "10,000+"
    context.company_enrichment.headquarters = "Munich"
    context.technology_stack.crm = [
        DetectedTechnology(name="Salesforce", confidence=0.9)
    ]
    context.leadership.leaders = [
        Leader(
            full_name="Alex Chief",
            job_title="CEO",
            organization="Example Co",
            source_url="https://example.com/leadership",
        ),
        Leader(
            full_name="Casey Finance",
            job_title="CFO",
            organization="Example Co",
            source_url="https://example.com/leadership",
        ),
        Leader(
            full_name="Sam Sales",
            job_title="VP Sales",
            organization="Example Co",
            source_url="https://example.com/leadership",
        ),
    ]
    context.business_signals.signals = [
        BusinessSignal(
            signal_type="funding",
            title="Raised a new round",
            description="Company funding event",
            source_url="https://example.com/news",
        )
    ]
    return context


def test_persona_signals_exclude_leadership_and_business_events() -> None:
    context = _visitor_context()

    signals = PersonaInference._context_signals(context)
    keys = {signal.key for signal in signals}
    descriptions = "\n".join(signal.description for signal in signals)

    assert not any(key.startswith("leadership.") for key in keys)
    assert not any(key.startswith("business_signal.") for key in keys)
    assert "Alex Chief" not in descriptions
    assert "Raised a new round" not in descriptions
    assert context.leadership.leaders  # still available to downstream stages


@pytest.mark.asyncio
async def test_changing_only_leadership_does_not_change_reasoning_request() -> None:
    first = _visitor_context()
    second = deepcopy(first)
    second.leadership.leaders[0].full_name = "Different Executive"
    second.leadership.leaders[0].job_title = "President and CEO"
    first_research = _CapturingResearch()
    second_research = _CapturingResearch()

    await PersonaInference(first_research)._infer(first)
    await PersonaInference(second_research)._infer(second)

    assert first_research.requests == second_research.requests


@pytest.mark.asyncio
async def test_reasoning_context_is_sectioned_and_behavior_drives_persona() -> None:
    research = _CapturingResearch()
    context = _visitor_context()

    decision, _ = await PersonaInference(research)._infer(context)

    assert decision is not None
    request = research.requests[0]
    assert [document.title for document in request.documents] == [
        "Behavioral Signals",
        "Company Context",
        "Technology Context",
    ]
    assert "CEO" not in "\n".join(document.content for document in request.documents)
    assert "anonymous website visitor" in request.instruction
    assert "contextual only" in request.instruction


@pytest.mark.asyncio
async def test_company_context_alone_cannot_support_a_persona() -> None:
    research = _CapturingResearch(
        {
            "personas": [
                {
                    "likely_persona": "Specific Employee",
                    "department": "Operations",
                    "seniority": None,
                    "reasoning": "Based only on company attributes.",
                    "supporting_signal_keys": [
                        "company.industry",
                        "company.company_size",
                    ],
                    "confidence": 0.7,
                }
            ],
            "overall_confidence": 0.7,
        }
    )

    decision, _ = await PersonaInference(research)._infer(_visitor_context())

    assert decision is None
