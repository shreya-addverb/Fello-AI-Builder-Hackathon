"""Regression tests for deterministic company identity resolution."""

from backend.research.models import ResearchDocument
from backend.services.company_identifier import CompanyIdentifier


def _documents() -> list[ResearchDocument]:
    return [
        ResearchDocument(
            source="web_search",
            title="Acme Corporation | Official Website",
            content="Acme Corporation provides enterprise software.",
            url="https://www.acme.com/about",
        ),
        ResearchDocument(
            source="official_website",
            title="About Acme Corporation",
            content="Acme Corporation is the company behind Acme enterprise software.",
            url="https://acme.com",
        ),
    ]


def test_aggregates_consistent_identity_candidate() -> None:
    candidates = CompanyIdentifier._aggregate_candidates(
        _documents(), "Acme Corp", None
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.canonical_name == "Acme Corp"
    assert "acme.com" in candidate.official_domains
    assert candidate.evidence_count == 2
    assert candidate.official_confirmations >= 1
    assert candidate.agreement == 1


def test_domain_normalization_handles_subdomains_and_country_domains() -> None:
    assert CompanyIdentifier._normalize_domain("https://www.investors.example.co.uk/path") == "investors.example.co.uk"
    assert CompanyIdentifier._registrable_domain("investors.example.co.uk") == "example.co.uk"
    assert CompanyIdentifier._registrable_domain("ir.example.com") == "example.com"


def test_organization_matching_supports_aliases_without_loose_substrings() -> None:
    assert CompanyIdentifier._organizations_match("IBM", "International Business Machines")
    assert CompanyIdentifier._organizations_match("Meta", "Meta Platforms Inc.")
    assert not CompanyIdentifier._organizations_match("Apple", "Apple Bank")
    assert not CompanyIdentifier._organizations_match("Oracle", "Oracle Cerner")


def test_deduplication_prefers_official_source_for_same_url() -> None:
    documents = [
        ResearchDocument(source="web_search", content="Acme company", url="https://acme.com"),
        ResearchDocument(source="official_website", content="Official Acme company page", url="https://acme.com/"),
    ]

    deduplicated = CompanyIdentifier._deduplicate_documents(documents)

    assert len(deduplicated) == 1
    assert deduplicated[0].source == "official_website"


def test_deterministic_decision_requires_official_agreement() -> None:
    candidate = CompanyIdentifier._aggregate_candidates(
        _documents(), "Acme Corp", None
    )[0]

    decision = CompanyIdentifier._decision_from_candidate(candidate)

    assert decision is not None
    assert decision.identified_company == "Acme Corp"
    assert decision.identified_domain == "acme.com"
    assert decision.supporting_urls
