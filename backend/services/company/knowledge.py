"""Centralized domain knowledge for company entity resolution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CompanyIdentificationKnowledge:
    excluded_domains: tuple[str, ...]
    legal_suffixes: frozenset[str]
    official_indicators: tuple[str, ...]
    two_level_suffixes: frozenset[str]
    source_quality: dict[str, float]
    aliases: dict[str, str]


KNOWLEDGE = CompanyIdentificationKnowledge(
    excluded_domains=(
        "wikipedia.org", "linkedin.com", "facebook.com", "instagram.com", "x.com",
        "youtube.com", "bloomberg.com", "reuters.com", "crunchbase.com",
    ),
    legal_suffixes=frozenset({
        "ag", "co", "company", "corp", "corporation", "gmbh", "group", "holdings",
        "inc", "incorporated", "limited", "llc", "ltd", "plc", "sa", "se",
    }),
    official_indicators=(
        "organization", "copyright", "privacy policy", "terms of use", "investor relations",
        "company registration", "legal", "open graph", "og:site_name", "about us", "official",
    ),
    two_level_suffixes=frozenset({"co.uk", "com.au", "co.in", "co.jp", "com.br", "com.sg"}),
    source_quality={
        "official_website": 1.0, "investor_relations": 0.95, "government": 0.92,
        "sec_filing": 0.92, "company_press_release": 0.88, "news": 0.68,
        "wikipedia": 0.55, "crunchbase": 0.5, "linkedin": 0.45, "web_search": 0.6,
    },
    aliases={"international business machines": "ibm", "meta platforms": "meta"},
)
