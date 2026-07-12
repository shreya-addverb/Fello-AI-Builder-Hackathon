"""Regression tests for evidence-grounded summary generation."""

from backend.models.context import (
    AnalysisContext,
    AnalysisInput,
    BusinessSignal,
    DetectedTechnology,
    InferredPersona,
    Intent,
    Leader,
)
from backend.services.summary_generation import (
    SummaryGeneration,
    _OpportunityDecision,
    _SummaryDecision,
    _SummaryFact,
    _SummarySection,
)


def _company_context() -> AnalysisContext:
    context = AnalysisContext(
        request_type="company",
        input=AnalysisInput(domain="example.com"),
    )
    context.company_identification.identified_company = "Example Co"
    context.company_identification.identified_domain = "example.com"
    context.company_enrichment.canonical_company_name = "Example Co"
    context.company_enrichment.industry = "Software"
    context.company_enrichment.business_description = (
        "Example Co provides analytics software for enterprise teams."
    )
    context.technology_stack.cloud = [
        DetectedTechnology(name="AWS", confidence=0.86)
    ]
    context.leadership.leaders = [
        Leader(
            full_name="Alex Leader",
            job_title="CEO",
            organization="Example Co",
            source_url="https://example.com/team",
        )
    ]
    context.business_signals.signals = [
        BusinessSignal(
            signal_type="product",
            title="Expanded analytics platform",
            description="The company released new enterprise analytics capabilities.",
            source_url="https://example.com/news",
        )
    ]
    context.persona.personas = [
        InferredPersona(
            likely_persona="Technical Evaluator",
            reasoning="Pricing and documentation activity indicate product evaluation.",
            supporting_signals=["visited pricing", "visited documentation"],
            confidence=0.76,
        )
    ]
    context.intent = Intent(
        intent_score=7,
        intent_stage="Evaluation",
        confidence=0.72,
        supporting_signals=["visited pricing", "returned this week"],
        reasoning_summary="Repeated pricing and documentation activity indicates evaluation.",
    )
    return context


def _section(*keys: str, text: str | None = "Supported section.") -> _SummarySection:
    return _SummarySection(text=text, supporting_fact_keys=list(keys))


def _decision(
    executive_keys: list[str],
    technology_keys: list[str] | None = None,
    opportunity_keys: list[str] | None = None,
    confidence: float = 0.9,
) -> _SummaryDecision:
    executive_text = (
        "Example Co is an enterprise software account with verified company, "
        "technology, leadership, business activity, persona, and intent evidence. "
        "The combined signals indicate an active account profile with current "
        "commercial context, technical footprint, and observable evaluation "
        "behavior suitable for an executive briefing."
    )
    return _SummaryDecision(
        executive_summary=_SummarySection(
            text=executive_text,
            supporting_fact_keys=executive_keys,
        ),
        company_overview=_section("company.name"),
        technology_overview=_section(*(technology_keys or ["technology.cloud.0"])),
        leadership_overview=_section("leadership.0"),
        business_activity_overview=_section("business_signal.0"),
        visitor_assessment=_section("persona.0", "intent.assessment"),
        buying_intent_assessment=_section("intent.assessment"),
        key_opportunities=[
            _OpportunityDecision(
                observation="Current evaluation activity is commercially relevant.",
                supporting_fact_keys=opportunity_keys
                or ["business_signal.0", "persona.0", "intent.assessment"],
            )
        ],
        confidence=confidence,
    )


def test_evidence_document_is_grouped_deduped_and_model_driven() -> None:
    facts = SummaryGeneration._context_facts(_company_context())
    facts.append(_SummaryFact(key="company.name", description="Example Co"))

    document = SummaryGeneration._format_evidence_document(facts)

    assert document.index("## COMPANY") < document.index("## TECHNOLOGY")
    assert document.index("## TECHNOLOGY") < document.index("## LEADERSHIP")
    assert "- technology.cloud.0 :: AWS; confidence=0.86" in document
    assert document.count("- company.name :: Example Co") == 1
    assert "\n\n## PERSONA\npersona:" in document
    assert "## BUYING INTENT\nintent:" in document


def test_summary_validation_rejects_semantically_wrong_citations() -> None:
    facts = SummaryGeneration._context_facts(_company_context())
    decision = _decision(
        executive_keys=[
            "company.name",
            "technology.cloud.0",
            "business_signal.0",
            "intent.assessment",
        ],
        technology_keys=["company.name"],
    )

    assert not SummaryGeneration._is_supported(decision, facts)


def test_executive_summary_requires_multiple_evidence_categories() -> None:
    facts = SummaryGeneration._context_facts(_company_context())
    decision = _decision(
        executive_keys=[
            "company.name",
            "company.industry",
            "company.description",
            "company.domain",
        ]
    )

    assert not SummaryGeneration._is_supported(decision, facts)


def test_context_model_uses_bounded_confidence() -> None:
    facts = SummaryGeneration._context_facts(_company_context())
    decision = _decision(
        executive_keys=[
            "company.name",
            "technology.cloud.0",
            "business_signal.0",
            "intent.assessment",
        ],
        confidence=1.0,
    )

    summary = SummaryGeneration._to_context_model(decision, facts)

    assert 0 < summary.confidence < 1.0
    assert summary.key_opportunities == [
        "Current evaluation activity is commercially relevant."
    ]


def test_identity_only_context_returns_insufficient_evidence_summary() -> None:
    context = AnalysisContext(
        request_type="company",
        input=AnalysisInput(company_name="Redfin", domain="redfin.com"),
    )
    context.company_identification.identified_company = "Redfin"
    context.company_identification.identified_domain = "redfin.com"

    facts = SummaryGeneration._context_facts(context)
    summary = SummaryGeneration._grounded_summary(context, facts)

    assert summary.confidence == 0
    assert summary.executive_summary is not None
    assert "has been identified as the account" in summary.executive_summary
    assert summary.key_opportunities
    assert summary.executive_summary != "redfin.com. Redfin."


def test_low_quality_company_only_enrichment_is_insufficient() -> None:
    context = AnalysisContext(
        request_type="company",
        input=AnalysisInput(company_name="Example Co", domain="example.com"),
    )
    context.company_identification.identified_company = "Example Co"
    context.company_identification.identified_domain = "example.com"
    context.company_enrichment.industry = "Software"
    context.company_enrichment.enrichment_confidence = 0.1

    facts = SummaryGeneration._context_facts(context)

    assert not SummaryGeneration._has_minimum_summary_evidence(context, facts)


def test_non_company_evidence_allows_grounded_summary() -> None:
    context = _company_context()
    facts = SummaryGeneration._context_facts(context)

    assert SummaryGeneration._has_minimum_summary_evidence(context, facts)
