"""Regression tests for pipeline orchestration semantics."""

import pytest

from backend.models.context import AnalysisContext, AnalysisInput
from backend.services.pipeline import AccountIntelligencePipeline


class _ResearchTrace:
    def __init__(self, entries=None):
        self.entries = entries or []

    @property
    def trace_count(self):
        return 0

    def trace_since(self, start):
        return self.entries


def _service(name, action=None, trace=None):
    async def execute(self, context):
        if action:
            action(context)
        return context

    service_type = type(name, (), {"execute": execute})
    instance = service_type()
    if trace is not None:
        instance._research = _ResearchTrace(trace)
    return instance


def _company_context() -> AnalysisContext:
    return AnalysisContext(
        request_type="company",
        input=AnalysisInput(domain="acme.com"),
    )


def _pipeline(identifier_action=None, summary_action=None):
    return AccountIntelligencePipeline(
        company_identifier=_service("CompanyIdentifier", identifier_action),
        company_enrichment=_service("CompanyEnrichment"),
        technology_detection=_service("TechnologyDetection"),
        leadership_discovery=_service("LeadershipDiscovery"),
        business_signals=_service("BusinessSignals"),
        persona_inference=_service("PersonaInference"),
        intent_scoring=_service("IntentScoring"),
        summary_generation=_service("SummaryGeneration", summary_action),
    )


@pytest.mark.asyncio
async def test_exception_is_recorded_and_later_summary_still_runs() -> None:
    def fail(_context):
        raise TypeError("broken stage")

    def summarize(context):
        context.ai_summary.executive_summary = "Partial account summary."

    context = await _pipeline(fail, summarize).execute(_company_context())

    identifier = context.pipeline_metadata.stage_results[0]
    assert identifier.status == "failed"
    assert "programming_error: TypeError" in (identifier.message or "")
    assert any("Traceback" in error for error in identifier.errors)
    assert "SummaryGeneration" in context.pipeline_metadata.completed_steps
    assert "CompanyEnrichment" not in context.pipeline_metadata.completed_steps
    assert context.pipeline_metadata.status == "failed"


@pytest.mark.asyncio
async def test_no_data_and_skipped_stages_are_not_completed_steps() -> None:
    def identify(context):
        context.company_identification.identified_domain = "acme.com"

    def summarize(context):
        context.ai_summary.executive_summary = "Account summary."

    context = await _pipeline(identify, summarize).execute(_company_context())
    results = {result.stage: result for result in context.pipeline_metadata.stage_results}

    assert results["CompanyIdentifier"].status == "completed"
    assert results["CompanyEnrichment"].status == "no_data"
    assert results["PersonaInference"].status == "skipped"
    assert results["IntentScoring"].status == "skipped"
    assert "CompanyEnrichment" not in context.pipeline_metadata.completed_steps
    assert "PersonaInference" not in context.pipeline_metadata.completed_steps
    assert context.pipeline_metadata.status == "completed"


def test_trace_processing_is_single_pass_and_malformed_safe() -> None:
    details = AccountIntelligencePipeline._trace_details([
        {"provider": "gemini", "model": "model-a", "retry_count": 1, "fallback": "cached", "error": None},
        {"provider": "gemini", "model": "model-a", "retry_count": "bad", "error": "timeout"},
        "malformed",
    ])

    assert details["providers"] == ["gemini"]
    assert details["models"] == ["model-a"]
    assert details["retries"] == 1
    assert details["fallbacks"] == ["cached"]
    assert details["errors"] == ["timeout", "Malformed research trace entry was ignored."]
