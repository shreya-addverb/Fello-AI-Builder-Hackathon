"""Account intelligence pipeline orchestration."""

import logging
import traceback
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable, ClassVar, Protocol

from pydantic import ValidationError

from backend.models.context import AnalysisContext, StageResult

logger = logging.getLogger(__name__)


class PipelineService(Protocol):
    """Interface implemented by every pipeline stage."""

    async def execute(self, context: AnalysisContext) -> AnalysisContext:
        """Execute a pipeline stage and return its context."""
        ...


class AccountIntelligencePipeline:
    """Run account intelligence services in their required order."""

    _OUTCOME_CHECKS: ClassVar[dict[str, Callable[[AnalysisContext], bool]]] = {
        "CompanyIdentifier": lambda context: bool(context.company_identification.identified_company or context.company_identification.identified_domain),
        "CompanyEnrichment": lambda context: bool(context.company_enrichment.canonical_company_name or context.company_enrichment.industry),
        "TechnologyDetection": lambda context: any(isinstance(value, list) and value for value in context.technology_stack.model_dump().values()),
        "LeadershipDiscovery": lambda context: bool(context.leadership.leaders),
        "BusinessSignals": lambda context: bool(context.business_signals.signals),
        "PersonaInference": lambda context: bool(context.persona.personas),
        "IntentScoring": lambda context: context.intent.intent_stage != "Unknown",
        "SummaryGeneration": lambda context: bool(context.ai_summary.executive_summary),
    }

    _COMPANY_CONTEXT_STAGES: ClassVar[frozenset[str]] = frozenset({
        "CompanyEnrichment", "TechnologyDetection", "LeadershipDiscovery", "BusinessSignals",
    })
    _OPTIONAL_RESEARCH_STAGES: ClassVar[frozenset[str]] = frozenset({
        "TechnologyDetection", "LeadershipDiscovery", "BusinessSignals",
    })

    def __init__(
        self,
        company_identifier: PipelineService,
        company_enrichment: PipelineService,
        technology_detection: PipelineService,
        leadership_discovery: PipelineService,
        business_signals: PipelineService,
        persona_inference: PipelineService,
        intent_scoring: PipelineService,
        summary_generation: PipelineService,
    ) -> None:
        """Receive all pipeline stages through dependency injection."""
        self._steps: tuple[PipelineService, ...] = (
            company_identifier,
            company_enrichment,
            technology_detection,
            leadership_discovery,
            business_signals,
            persona_inference,
            intent_scoring,
            summary_generation,
        )

    async def execute(self, context: AnalysisContext) -> AnalysisContext:
        """Execute all stages sequentially and return application state."""
        context.pipeline_metadata.status = "running"
        context.pipeline_metadata.completed_steps.clear()
        context.pipeline_metadata.stage_results.clear()

        for service in self._steps:
            stage = service.__class__.__name__
            should_execute, skip_message = self._should_execute(stage, context)
            started_at = datetime.now(timezone.utc)
            started = perf_counter()
            research = getattr(service, "_research", None)
            trace_start = self._trace_count(research)
            error_messages: list[str] = []
            if not should_execute:
                status, message = "skipped", skip_message
            else:
                try:
                    context = await service.execute(context)
                    status, message = self._stage_outcome(stage, context)
                    if status == "completed":
                        context.pipeline_metadata.completed_steps.append(stage)
                except Exception as error:
                    category = self._exception_category(error)
                    summary = f"{category}: {type(error).__name__}: {error}"
                    status, message = "failed", summary
                    formatted_traceback = traceback.format_exc()
                    error_messages.extend((summary, formatted_traceback))
                    logger.exception("Pipeline stage %s failed (%s).", stage, category)
            trace = self._trace_since(research, trace_start)
            trace_details = self._trace_details(trace)
            if (
                should_execute
                and status == "no_data"
                and trace_details["errors"]
                and stage not in self._OPTIONAL_RESEARCH_STAGES
            ):
                status = "failed"
                message = "Provider failure prevented the stage from producing supported data."
            elif status == "failed" and stage in self._OPTIONAL_RESEARCH_STAGES:
                status = "no_data"
                message = (
                    "Optional research could not produce supported data; continuing with "
                    "available account intelligence."
                )
            ended = perf_counter()
            ended_at = datetime.now(timezone.utc)
            context.pipeline_metadata.stage_results.append(
                StageResult(
                    stage=stage,
                    status=status,
                    duration_ms=max(0, round((ended - started) * 1000)),
                    message=message,
                    started_at=started_at,
                    ended_at=ended_at,
                    providers_used=trace_details["providers"],
                    models_used=trace_details["models"],
                    retry_count=trace_details["retries"],
                    fallback_events=trace_details["fallbacks"],
                    errors=error_messages + trace_details["errors"],
                )
            )
            if stage == "SummaryGeneration" and status != "failed":
                has_recommendations = bool(context.sales_recommendations.recommended_actions)
                context.pipeline_metadata.stage_results.append(
                    StageResult(
                        stage="SalesRecommendationGeneration",
                        status="completed" if has_recommendations else "no_data",
                        duration_ms=0,
                        message=(
                            "Completed successfully."
                            if has_recommendations
                            else "Stage completed successfully but produced no supported data."
                        ),
                        started_at=ended_at,
                        ended_at=ended_at,
                    )
                )
                if has_recommendations:
                    context.pipeline_metadata.completed_steps.append("SalesRecommendationGeneration")

        results = context.pipeline_metadata.stage_results
        context.pipeline_metadata.status = (
            "failed"
            if any(result.status == "failed" for result in results)
            or not context.pipeline_metadata.completed_steps
            else "completed"
        )
        return context

    @staticmethod
    def _stage_outcome(stage: str, context: AnalysisContext) -> tuple[str, str | None]:
        checker = AccountIntelligencePipeline._OUTCOME_CHECKS.get(stage)
        if checker is None or checker(context):
            return "completed", "Completed successfully."
        return "no_data", "Stage completed successfully but produced no supported data."

    @classmethod
    def _should_execute(
        cls,
        stage: str,
        context: AnalysisContext,
    ) -> tuple[bool, str | None]:
        if stage in {"PersonaInference", "IntentScoring"} and context.request_type == "company":
            return False, "Skipped because the stage is not applicable to company-only analysis."
        identity_available = bool(
            context.company_identification.identified_company
            or context.company_identification.identified_domain
        )
        enrichment_available = bool(
            context.company_enrichment.canonical_company_name
            or context.company_enrichment.website
            or context.company_enrichment.industry
        )
        if stage == "CompanyEnrichment" and not identity_available:
            return False, "Skipped because company identification was unavailable."
        if stage in cls._COMPANY_CONTEXT_STAGES - {"CompanyEnrichment"} and not (
            identity_available or enrichment_available
        ):
            return False, "Skipped because company identity and enrichment prerequisites were unavailable."
        if stage in {"PersonaInference", "IntentScoring"} and not cls._behavior_available(context):
            return False, "Skipped because visitor behavioral prerequisites were unavailable."
        return True, None

    @staticmethod
    def _behavior_available(context: AnalysisContext) -> bool:
        input_data = context.input
        return bool(input_data.pages_visited) or any(
            value is not None
            for value in (
                input_data.visit_duration,
                input_data.visits_this_week,
                input_data.referral_source,
                input_data.visit_timestamp,
            )
        )

    @staticmethod
    def _exception_category(error: Exception) -> str:
        if isinstance(error, ValidationError):
            return "validation_failure"
        if isinstance(error, (TypeError, AttributeError, AssertionError, KeyError)):
            return "programming_error"
        if isinstance(error, (TimeoutError, ConnectionError)) or error.__class__.__module__.startswith("httpx"):
            return "provider_failure"
        return "unexpected_runtime_error"

    @staticmethod
    def _trace_count(research: Any) -> int:
        value = getattr(research, "trace_count", 0)
        return value if isinstance(value, int) and value >= 0 else 0

    @staticmethod
    def _trace_since(research: Any, start: int) -> list[Any]:
        method = getattr(research, "trace_since", None)
        if not callable(method):
            return []
        try:
            trace = method(start)
        except Exception as error:
            logger.warning("Failed to collect research trace (%s).", type(error).__name__)
            return []
        return trace if isinstance(trace, list) else []

    @staticmethod
    def _trace_details(trace: list[Any]) -> dict[str, Any]:
        providers: list[str] = []
        models: list[str] = []
        fallbacks: list[str] = []
        errors: list[str] = []
        seen_providers: set[str] = set()
        seen_models: set[str] = set()
        retries = 0
        for item in trace:
            if not isinstance(item, dict):
                errors.append("Malformed research trace entry was ignored.")
                continue
            provider = item.get("provider")
            if isinstance(provider, str) and provider and provider not in seen_providers:
                seen_providers.add(provider)
                providers.append(provider)
            model = item.get("model")
            if isinstance(model, str) and model and model not in seen_models:
                seen_models.add(model)
                models.append(model)
            retry_count = item.get("retry_count", 0)
            if isinstance(retry_count, int) and retry_count > 0:
                retries += retry_count
            fallback = item.get("fallback")
            if isinstance(fallback, str) and fallback:
                fallbacks.append(fallback)
            error = item.get("error")
            if isinstance(error, str) and error:
                errors.append(error)
        return {
            "providers": providers,
            "models": models,
            "retries": retries,
            "fallbacks": fallbacks,
            "errors": errors,
        }
