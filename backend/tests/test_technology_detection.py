"""Regression tests for evidence-grounded technology detection."""

import pytest
from pydantic import ValidationError

from backend.models.context import AnalysisContext, AnalysisInput
from backend.research.models import (
    CrawlMetadata,
    CrawlResponse,
    ReasonResponse,
    ResearchDocument,
    SearchResponse,
)
from backend.services.technology_detection import (
    TechnologyDetection,
    _DetectionDecision,
    _TechnologyDecision,
    _TechnologyEvidence,
)


OFFICIAL_URL = "https://acme.com/engineering"
SEARCH_URL = "https://news.example/acme-stack"


def _document(
    source: str = "official_engineering",
    url: str = OFFICIAL_URL,
    content: str = "Acme uses React for its customer portal.",
) -> ResearchDocument:
    return ResearchDocument(
        source=source,
        title="Acme engineering",
        content=content,
        url=url,
    )


def _technology(
    name: str = "React",
    url: str = OFFICIAL_URL,
    indicator: str = "Acme uses React for its customer portal.",
    confidence: float = 0.95,
) -> _TechnologyDecision:
    return _TechnologyDecision(
        name=name,
        category="frontend",
        confidence=confidence,
        evidence=[_TechnologyEvidence(url=url, indicator=indicator)],
    )


def test_evidence_profile_deduplicates_and_groups_documents() -> None:
    profile = TechnologyDetection._evidence_profile(
        [
            _document(url="https://www.acme.com/engineering?utm_source=test"),
            _document(url="https://acme.com/engineering"),
            _document(source="technology_search", url=SEARCH_URL),
        ]
    )
    evidence_document = TechnologyDetection._evidence_document(profile)

    assert profile.total_documents == 2
    assert len(profile.duplicates) == 1
    assert profile.source_groups == {"official", "search"}
    assert "## DOCUMENTATION" in evidence_document.content
    assert "## SEARCH RESULTS" in evidence_document.content
    assert "utm_source" not in evidence_document.content


def test_evidence_normalization_removes_boilerplate_and_tracking() -> None:
    profile = TechnologyDetection._evidence_profile(
        [
            _document(
                url="https://www.acme.com/engineering?utm_campaign=x&ref=keep",
                content="  Cookie preferences\nAcme uses React for its portal.\nAll rights reserved. ",
            )
        ]
    )

    assert profile.records[0].normalized_url == "https://acme.com/engineering?ref=keep"
    assert "Cookie preferences" not in profile.records[0].document.content
    assert profile.records[0].temporal_status == "current"


def test_validation_requires_meaningful_supported_indicators() -> None:
    profile = TechnologyDetection._evidence_profile([_document()])

    supported = _DetectionDecision(
        technologies=[_technology()],
        overall_confidence=0.9,
    )
    name_only = _DetectionDecision(
        technologies=[_technology(indicator="React")],
        overall_confidence=0.9,
    )
    missing_url = _DetectionDecision(
        technologies=[_technology(url="https://missing.example/")],
        overall_confidence=0.9,
    )

    assert TechnologyDetection._is_supported(supported, profile)
    assert not TechnologyDetection._is_supported(name_only, profile)
    assert not TechnologyDetection._is_supported(missing_url, profile)


def test_validation_rejects_unsupported_and_duplicate_evidence_urls() -> None:
    profile = TechnologyDetection._evidence_profile([_document()])
    duplicate_url = _TechnologyDecision(
        name="React",
        category="frontend",
        confidence=0.9,
        evidence=[
            _TechnologyEvidence(
                url=OFFICIAL_URL,
                indicator="Acme uses React for its customer portal.",
            ),
            _TechnologyEvidence(
                url="https://www.acme.com/engineering?utm_source=test",
                indicator="Acme uses React for its customer portal.",
            ),
        ],
    )
    unsupported = _DetectionDecision(
        technologies=[_technology(indicator="Acme likes Vue for examples.")],
        overall_confidence=0.9,
    )

    assert not TechnologyDetection._is_supported(
        _DetectionDecision(technologies=[duplicate_url], overall_confidence=0.9),
        profile,
    )
    assert not TechnologyDetection._is_supported(unsupported, profile)


def test_duplicate_detection_normalizes_punctuation_and_case() -> None:
    with pytest.raises(ValidationError):
        _DetectionDecision(
            technologies=[
                _technology(name="React.js"),
                _technology(name="react js", indicator="Acme uses react js."),
            ],
            overall_confidence=0.8,
        )


def test_conflicting_evidence_is_tracked_and_lowers_confidence() -> None:
    profile = TechnologyDetection._evidence_profile(
        [
            _document(content="Acme uses React for its customer portal."),
            _document(
                source="technology_search",
                url=SEARCH_URL,
                content="Acme previously uses React in the old customer portal.",
            ),
        ]
    )
    decision = _DetectionDecision(
        technologies=[_technology(confidence=1.0)],
        overall_confidence=1.0,
    )

    bounded = TechnologyDetection._bounded_decision(decision, profile)

    assert "react" in profile.conflicts
    assert profile.historical_indicators
    assert bounded.overall_confidence < 0.8


def test_confidence_is_bounded_by_evidence_quality() -> None:
    profile = TechnologyDetection._evidence_profile(
        [
            _document(),
            _document(source="technology_search", url=SEARCH_URL),
        ]
    )
    decision = _DetectionDecision(
        technologies=[_technology(confidence=1.0)],
        overall_confidence=1.0,
    )

    bounded = TechnologyDetection._bounded_decision(decision, profile)

    assert 0 < bounded.technologies[0].confidence < 1.0
    assert 0 < bounded.overall_confidence < 1.0


def test_crawl_targets_are_deterministic_and_generic() -> None:
    targets = TechnologyDetection._crawl_targets("https://www.acme.com/")

    assert targets[0] == ("https://acme.com/", "official_homepage")
    assert len(targets) == len({url for url, _source in targets})
    assert any(source == "official_docs" for _url, source in targets)


class _FallbackResearch:
    async def search(self, _request):
        return SearchResponse(provider="test", results=[], duration_ms=0)

    async def crawl(self, request):
        return CrawlResponse(
            provider="test",
            markdown="Acme uses React for its customer portal.",
            metadata=CrawlMetadata(title="Acme engineering", source_url=request.url),
            duration_ms=0,
        )

    async def reason(self, _request):
        return ReasonResponse(provider="test", structured_output=None, duration_ms=0)


@pytest.mark.asyncio
async def test_reasoning_failure_uses_conservative_fallback() -> None:
    context = AnalysisContext(
        request_type="company",
        input=AnalysisInput(domain="acme.com"),
    )
    context.company_identification.identified_domain = "acme.com"

    result = await TechnologyDetection(_FallbackResearch()).execute(context)

    assert result.technology_stack.other
    assert result.technology_stack.other[0].name == "React"
    assert result.technology_stack.detection_confidence > 0
