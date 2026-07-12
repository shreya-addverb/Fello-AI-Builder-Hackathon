"""Safe identity preservation when external company identification is unavailable."""

from urllib.parse import urlsplit

from backend.models.context import AnalysisInput, CompanyIdentification


def build_company_identification(input_data: AnalysisInput) -> CompanyIdentification:
    """Preserve only user-supplied identity values without inventing company facts."""
    domain = _normalize_domain(input_data.domain)
    return CompanyIdentification(
        identified_company=input_data.company_name,
        identified_domain=domain,
        identification_confidence=0.5 if input_data.company_name or domain else 0.0,
    )


def _normalize_domain(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip().lower()
    if "://" in candidate:
        candidate = urlsplit(candidate).netloc
    candidate = candidate.split("/")[0].removeprefix("www.").rstrip(".")
    return candidate or None
