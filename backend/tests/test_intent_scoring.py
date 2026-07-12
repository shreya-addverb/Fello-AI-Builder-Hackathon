"""Regression tests for conservative evidence-based intent scoring."""

from backend.models.context import AnalysisContext, AnalysisInput, InferredPersona
from backend.services.intent_scoring import (
    IntentScoring,
    _IntentDecision,
    _SignalCategory,
)


def _visitor() -> AnalysisContext:
    return AnalysisContext(
        request_type="visitor",
        input=AnalysisInput(
            visitor_id="visitor-1",
            ip_address="8.8.8.8",
            pages_visited=["/pricing", "/docs/api", "/integrations"],
            visit_duration=420,
            visits_this_week=3,
            referral_source="search",
        ),
    )


def _decision(**updates) -> _IntentDecision:
    values = {
        "intent_score": 7,
        "intent_stage": "Evaluation",
        "confidence": 0.65,
        "supporting_signal_keys": [
            "behavior.page_type.pricing",
            "persona.role.technical_evaluator.0",
        ],
        "reasoning_summary": "Pricing activity and technical evaluator alignment support active product evaluation.",
    }
    values.update(updates)
    return _IntentDecision(**values)


def test_context_builds_weighted_derived_behavior_signals() -> None:
    context = _visitor()

    signals = IntentScoring._context_signals(context)
    by_key = {signal.key: signal for signal in signals}

    assert "behavior.page_type.pricing" in by_key
    assert "behavior.page_type.documentation" in by_key
    assert "behavior.page_type.integration" in by_key
    assert "behavior.engagement_score" in by_key
    assert by_key["behavior.engagement_score"].category == _SignalCategory.BEHAVIOR
    assert by_key["behavior.engagement_score"].weight == 1


def test_rejects_stage_score_mismatch() -> None:
    assert not IntentScoring._validate_stage_score(
        _decision(intent_stage="Decision", intent_score=3)
    )
    assert not IntentScoring._validate_stage_score(
        _decision(intent_stage="Awareness", intent_score=9)
    )


def test_known_intent_requires_two_signal_categories() -> None:
    context = _visitor()
    signals = IntentScoring._context_signals(context)
    behavior_keys = [signal.key for signal in signals if signal.category == _SignalCategory.BEHAVIOR][:2]
    decision = _decision(
        intent_stage="Research",
        intent_score=3,
        confidence=0.4,
        supporting_signal_keys=behavior_keys,
        reasoning_summary="Visited pages and sustained website activity show continued product research behavior.",
    )

    assert not IntentScoring._is_supported(decision, signals, context)


def test_pricing_with_short_visit_is_conflicting_evidence() -> None:
    context = _visitor()
    context.input.visit_duration = 5

    assert "pricing_with_very_short_visit" in IntentScoring._detect_signal_conflicts(context)


def test_decision_rejected_for_end_user_only_persona() -> None:
    context = _visitor()
    context.persona.personas = [
        InferredPersona(
            likely_persona="End User Analyst",
            reasoning="Behavior supports an end-user persona.",
            supporting_signals=["Pricing page", "Documentation page"],
            confidence=0.8,
        )
    ]

    assert IntentScoring._only_end_user_personas(context)
