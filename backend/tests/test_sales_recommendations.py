"""Regression tests for sales recommendation guardrails."""

from backend.models.context import AnalysisContext, AnalysisInput, Leader
from backend.services.sales_recommendations import SalesRecommendationGeneration


def test_leadership_only_context_does_not_recommend_executive_outreach() -> None:
    context = AnalysisContext(
        request_type="company",
        input=AnalysisInput(company_name="Siemens", domain="siemens.com"),
    )
    context.company_identification.identified_company = "Siemens"
    context.company_identification.identified_domain = "siemens.com"
    context.company_enrichment.canonical_company_name = "Siemens"
    context.company_enrichment.website = "https://siemens.com/"
    context.leadership.leaders = [
        Leader(
            full_name="Ann Fairchild",
            job_title="President & CEO, Siemens USA",
            organization="Siemens",
            source_url="https://www.siemens.com/en-us/company/leadership/us-management",
            confidence=0.95,
        )
    ]

    recommendations = SalesRecommendationGeneration._grounded_recommendations(context)

    assert recommendations.account_priority == "LOW"
    assert recommendations.urgency == "LOW"
    assert recommendations.recommended_contacts == []
    assert any(
        "clearer business trigger" in action
        for action in recommendations.recommended_actions
    )
