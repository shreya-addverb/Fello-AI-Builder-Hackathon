"""Focused correctness tests for evidence-backed leadership discovery."""

from backend.research.models import ResearchDocument
from backend.services.leadership_discovery import LeadershipDiscovery


SOURCE_URL = "https://example.com/leadership"


def _document() -> ResearchDocument:
    return ResearchDocument(
        source="official_leadership",
        url=SOURCE_URL,
        content=(
            "Roland Busch, President and Chief Executive Officer. "
            "Patricia Thomas, Chief Marketing Officer."
        ),
    )


def _leader(name: str, title: str, indicator: str, **overrides):
    leader = {
        "full_name": name,
        "job_title": title,
        "department": None,
        "organization": "Example Company",
        "linkedin_url": None,
        "source_url": SOURCE_URL,
        "confidence": 0.9,
        "evidence_indicator": indicator,
    }
    leader.update(overrides)
    return leader


def test_rejects_surname_only_and_keeps_valid_full_name() -> None:
    decision = LeadershipDiscovery._validated_decision(
        {
            "leaders": [
                _leader("Busch", "CEO", "Busch President and Chief Executive Officer"),
                _leader(
                    "Roland Busch",
                    "CEO",
                    "Roland Busch President and Chief Executive Officer",
                ),
            ],
            "overall_confidence": 0.9,
        },
        [_document()],
        "Example Company",
    )

    assert decision is not None
    assert [leader.full_name for leader in decision.leaders] == ["Roland Busch"]


def test_rejects_abbreviation_not_used_by_official_evidence() -> None:
    decision = LeadershipDiscovery._validated_decision(
        {
            "leaders": [
                _leader(
                    "R. Busch",
                    "CEO",
                    "Roland Busch President and Chief Executive Officer",
                )
            ],
            "overall_confidence": 0.9,
        },
        [_document()],
        "Example Company",
    )

    assert decision is not None
    assert decision.leaders == []
    assert decision.overall_confidence == 0


def test_merges_duplicate_person_and_keeps_stronger_title() -> None:
    decision = LeadershipDiscovery._validated_decision(
        {
            "leaders": [
                _leader(
                    "Roland Busch",
                    "CEO",
                    "Roland Busch President and Chief Executive Officer",
                    confidence=0.88,
                ),
                _leader(
                    "Roland Busch",
                    "President and Chief Executive Officer",
                    "Roland Busch President and Chief Executive Officer",
                    confidence=0.94,
                ),
            ],
            "overall_confidence": 0.94,
        },
        [_document()],
        "Example Company",
    )

    assert decision is not None
    assert len(decision.leaders) == 1
    assert decision.leaders[0].job_title == "President and Chief Executive Officer"
    assert decision.leaders[0].confidence == 0.94


def test_discards_wrong_organization_and_inflated_weak_source_confidence() -> None:
    weak_document = _document().model_copy(update={"source": "leadership_news"})
    decision = LeadershipDiscovery._validated_decision(
        {
            "leaders": [
                _leader(
                    "Roland Busch",
                    "CEO",
                    "Roland Busch President and Chief Executive Officer",
                    organization="Unrelated Corporation",
                ),
                _leader(
                    "Patricia Thomas",
                    "CMO",
                    "Patricia Thomas Chief Marketing Officer",
                    confidence=0.96,
                ),
            ],
            "overall_confidence": 0.96,
        },
        [weak_document],
        "Example Company",
    )

    assert decision is not None
    assert decision.leaders == []


def test_prompt_prioritizes_complete_verified_names_and_partial_success() -> None:
    instruction = LeadershipDiscovery._reasoning_instruction("Example Company")

    assert "complete publicly listed name" in instruction
    assert "Correctness is more important than recall" in instruction
    assert "omit the person" in instruction
