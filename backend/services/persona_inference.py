"""Evidence-based Persona Inference pipeline stage."""

from time import perf_counter
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from backend.models.context import AnalysisContext, InferredPersona, Persona
from backend.research.models import ReasonRequest, ResearchDocument
from backend.research.service import ResearchService
from backend.utils.service_logging import (
    log_execution_completed,
    log_execution_started,
    log_provider_failure,
)


class _AvailableSignal(BaseModel):
    """A typed context signal available to the reasoning engine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1)
    description: str = Field(min_length=1)


class _PersonaDecision(BaseModel):
    """Internal persona assessment returned by Gemini."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    likely_persona: str = Field(min_length=1)
    department: str | None = Field(default=None, min_length=1)
    seniority: str | None = Field(default=None, min_length=1)
    reasoning: str = Field(min_length=1)
    supporting_signal_keys: list[str] = Field(min_length=2)
    confidence: float = Field(ge=0, le=1)


class _PersonaInferenceDecision(BaseModel):
    """Validated structured persona decision returned by Gemini."""

    model_config = ConfigDict(extra="forbid")

    personas: list[_PersonaDecision] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_decision(self) -> "_PersonaInferenceDecision":
        """Reject duplicate personas and nonzero confidence for an empty result."""
        identities = [
            (
                persona.likely_persona.casefold(),
                (persona.department or "").casefold(),
                (persona.seniority or "").casefold(),
            )
            for persona in self.personas
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate persona assessments are not allowed.")
        if not self.personas and self.overall_confidence != 0:
            raise ValueError("Empty persona results must have zero confidence.")
        return self


class PersonaInference:
    """Infer likely visitor personas from existing pipeline evidence."""

    _persona_schema: ClassVar[JsonValue] = {
        "type": "object",
        "properties": {
            "personas": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "likely_persona": {"type": "string"},
                        "department": {"type": ["string", "null"]},
                        "seniority": {"type": ["string", "null"]},
                        "reasoning": {"type": "string"},
                        "supporting_signal_keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": [
                        "likely_persona",
                        "department",
                        "seniority",
                        "reasoning",
                        "supporting_signal_keys",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            },
            "overall_confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "required": ["personas", "overall_confidence"],
        "additionalProperties": False,
    }

    def __init__(self, research_service: ResearchService) -> None:
        """Receive the shared reasoning facade through dependency injection."""
        self._research = research_service

    async def execute(self, context: AnalysisContext) -> AnalysisContext:
        """Populate only the visitor persona section of the context."""
        service_name = self.__class__.__name__
        started_at = perf_counter()
        log_execution_started(service_name, "research_service")

        try:
            decision, signals = await self._infer(context)
        except Exception:
            log_provider_failure(service_name, "research_service")
            decision, signals = None, []

        if context.request_type == "company":
            context.persona = Persona()
        elif decision is None:
            context.persona = Persona()
        else:
            context.persona = self._to_context_model(decision, signals)
        duration_ms = max(0, round((perf_counter() - started_at) * 1000))
        succeeded = decision is not None and bool(decision.personas)
        log_execution_completed(service_name, succeeded, duration_ms)
        return context

    async def _infer(
        self,
        context: AnalysisContext,
    ) -> tuple[_PersonaInferenceDecision | None, list[_AvailableSignal]]:
        if context.request_type != "visitor":
            return None, []

        signals = self._context_signals(context)
        if len(signals) < 2:
            return None, signals

        documents = self._sectioned_documents(signals)
        response = await self._research.reason(
            ReasonRequest(
                instruction=self._reasoning_instruction(),
                documents=documents,
                output_mode="json",
                json_schema=self._persona_schema,
            )
        )
        if not response.succeeded or response.structured_output is None:
            return None, signals

        try:
            decision = _PersonaInferenceDecision.model_validate(
                response.structured_output
            )
            return (
                (decision, signals)
                if self._is_supported(decision, signals)
                else (None, signals)
            )
        except (TypeError, ValueError):
            return None, signals

    @staticmethod
    def _context_signals(context: AnalysisContext) -> list[_AvailableSignal]:
        signals: list[_AvailableSignal] = []
        input_data = context.input

        for index, page in enumerate(input_data.pages_visited):
            signals.append(
                _AvailableSignal(
                    key=f"behavior.page.{index}",
                    description=f"Visited page: {page}",
                )
            )
        if input_data.visit_duration is not None:
            signals.append(
                _AvailableSignal(
                    key="behavior.visit_duration",
                    description=f"Visit duration seconds: {input_data.visit_duration}",
                )
            )
        if input_data.visits_this_week is not None:
            signals.append(
                _AvailableSignal(
                    key="behavior.visit_frequency",
                    description=f"Visits this week: {input_data.visits_this_week}",
                )
            )
        if input_data.referral_source is not None:
            signals.append(
                _AvailableSignal(
                    key="behavior.referral_source",
                    description=f"Referral source: {input_data.referral_source}",
                )
            )
        if input_data.device_type is not None:
            signals.append(
                _AvailableSignal(
                    key="behavior.device_type",
                    description=f"Device type: {input_data.device_type}",
                )
            )

        enrichment = context.company_enrichment
        for key, value in (
            ("company.industry", enrichment.industry),
            ("company.business_category", enrichment.business_category),
            ("company.company_size", enrichment.company_size),
            ("company.headquarters", enrichment.headquarters),
        ):
            if value is not None:
                signals.append(_AvailableSignal(key=key, description=str(value)))

        technology_categories = (
            ("crm", context.technology_stack.crm),
            ("marketing", context.technology_stack.marketing),
            ("analytics", context.technology_stack.analytics),
            ("cms", context.technology_stack.cms),
            ("frontend", context.technology_stack.frontend),
            ("backend", context.technology_stack.backend),
            ("hosting", context.technology_stack.hosting),
            ("cloud", context.technology_stack.cloud),
            ("security", context.technology_stack.security),
            ("databases", context.technology_stack.databases),
            ("ai_platforms", context.technology_stack.ai_platforms),
            ("developer_tools", context.technology_stack.developer_tools),
            ("customer_support", context.technology_stack.customer_support),
            ("other", context.technology_stack.other),
        )
        for category, technologies in technology_categories:
            for index, technology in enumerate(technologies):
                signals.append(
                    _AvailableSignal(
                        key=f"technology.{category}.{index}",
                        description=(
                            f"{technology.name}; confidence={technology.confidence}"
                        ),
                    )
                )

        return signals

    @staticmethod
    def _sectioned_documents(
        signals: list[_AvailableSignal],
    ) -> list[ResearchDocument]:
        """Keep visitor evidence semantically separate from company context."""
        sections = (
            ("behavior.", "Behavioral Signals", "visitor_behavior"),
            ("company.", "Company Context", "company_context"),
            ("technology.", "Technology Context", "technology_context"),
        )
        documents: list[ResearchDocument] = []
        for prefix, title, source in sections:
            section_signals = [
                signal for signal in signals if signal.key.startswith(prefix)
            ]
            if section_signals:
                documents.append(
                    ResearchDocument(
                        source=source,
                        title=title,
                        content=(
                            f"{title}\n"
                            + "\n".join(
                                f"{signal.key}: {signal.description}"
                                for signal in section_signals
                            )
                        ),
                    )
                )
        return documents

    @staticmethod
    def _reasoning_instruction() -> str:
        return (
            "You are an AI system performing visitor persona inference for an Account Intelligence platform.\n\n"

        "Your task is to infer the MOST LIKELY ROLE of an anonymous website visitor based ONLY on the evidence provided.\n\n"

        "IMPORTANT:\n"
        "The visitor has NOT been identified.\n"
        "You are NOT identifying a person.\n"
        "You are estimating the visitor's likely professional role based on browsing behaviour.\n\n"

        "The supplied evidence is divided into semantic sections.\n"
        "Treat each section differently.\n\n"

        "Behavioral Signals:\n"
        "- These are the PRIMARY evidence.\n"
        "- They include page sequence, pages visited, visit duration, repeat visits, referral information, campaigns, and other browsing behaviour.\n"
        "- Behaviour should dominate every inference.\n\n"

        "Company Context:\n"
        "- This is SECONDARY evidence.\n"
        "- Industry, company size, headquarters and business category only provide context.\n"
        "- Company context MAY help resolve ambiguity or slightly adjust confidence.\n"
        "- Company context MUST NEVER create a persona by itself.\n\n"

        "Technology Context:\n"
        "- Technology stack is contextual only.\n"
        "- Technologies may refine an already-supported behavioural inference.\n"
        "- Technology alone MUST NEVER justify a persona.\n\n"

        "Reasoning Rules:\n"
        "- Infer personas only from behavioural evidence.\n"
        "- Behaviour must always outweigh company context.\n"
        "- Behaviour must always outweigh technology context.\n"
        "- Prefer evidence over assumptions.\n"
        "- When evidence is weak, return fewer personas.\n"
        "- When evidence is insufficient, return an empty persona list.\n"
        "- Never guess.\n"
        "- Never hallucinate.\n\n"

        "Leadership Rules:\n"
        "- Company leadership is NOT the visitor.\n"
        "- Never infer that the visitor is the CEO, CFO, Founder, President, VP, Director, Board Member, Executive, Manager, or any other specific employee simply because those people belong to the researched company.\n"
        "- Named people discovered elsewhere in the pipeline are company contacts for future outreach only.\n"
        "- They are NEVER evidence of visitor identity.\n\n"

        "Business Signal Rules:\n"
        "- Hiring, funding, acquisitions, expansion, partnerships, product launches and other company events describe the company, not the visitor.\n"
        "- These signals must NEVER influence visitor persona.\n\n"

        "Visitor Journey:\n"
        "- Consider the order of pages visited.\n"
        "- A navigation path often provides stronger evidence than isolated page views.\n"
        "- Consider how the visitor progressed through the website before drawing conclusions.\n\n"

        "Confidence:\n"
        "- Confidence must be proportional to evidence quality.\n"
        "- High confidence requires multiple independent behavioural signals pointing toward the same role.\n"
        "- Sparse, conflicting or ambiguous evidence should reduce confidence.\n"
        "- Confidence greater than 0.90 should be rare and only used when behavioural evidence is exceptionally strong.\n\n"

        "Multiple Personas:\n"
        "- Prefer one primary persona whenever possible.\n"
        "- Return multiple personas only when multiple independent behavioural patterns genuinely support them.\n"
        "- Do not create multiple personas simply because several departments could plausibly exist.\n\n"

        "Reasoning:\n"
        "- Explain your conclusion using observable evidence only.\n"
        "- Do not speculate.\n"
        "- Do not expose hidden reasoning.\n"
        "- Keep reasoning concise (1-3 sentences).\n"
        "- Reference only the supplied signal keys.\n\n"

        "Validation:\n"
        "- Every persona must reference at least two supporting signal keys.\n"
        "- At least one supporting key MUST be behavioural.\n"
        "- Supporting keys MUST exist in the supplied evidence.\n"
        "- Do not invent evidence.\n\n"

        "Examples of possible personas include (but are not limited to):\n"
        "Sales Operations, Revenue Operations, Marketing, Engineering, Solutions Architect, Procurement, IT, Security, Product Management, Customer Success, Finance, Commercial Operations, Developer Relations, Innovation.\n"
        "These are examples only. Do not treat them as an exhaustive or preferred list.\n\n"

        "Your objective is to produce conservative, evidence-based, explainable persona inference. "
        "It is always preferable to return no persona than to fabricate one."
    )

    @staticmethod
    def _is_supported(
        decision: _PersonaInferenceDecision,
        signals: list[_AvailableSignal],
    ) -> bool:
        available_keys = {signal.key for signal in signals}
        for persona in decision.personas:
            keys = persona.supporting_signal_keys
            if len(keys) != len(set(keys)):
                return False
            if len(keys) < 2 or not set(keys).issubset(available_keys):
                return False
            if not any(key.startswith("behavior.") for key in keys):
                return False
        return True

    @staticmethod
    def _to_context_model(
        decision: _PersonaInferenceDecision | None,
        signals: list[_AvailableSignal],
    ) -> Persona:
        if decision is None:
            return Persona(overall_persona_confidence=0.0)
        descriptions = {signal.key: signal.description for signal in signals}
        return Persona(
            personas=[
                InferredPersona(
                    likely_persona=persona.likely_persona,
                    department=persona.department,
                    seniority=persona.seniority,
                    reasoning=persona.reasoning,
                    supporting_signals=[
                        descriptions[key] for key in persona.supporting_signal_keys
                    ],
                    confidence=persona.confidence,
                )
                for persona in decision.personas
            ],
            overall_persona_confidence=decision.overall_confidence,
        )
