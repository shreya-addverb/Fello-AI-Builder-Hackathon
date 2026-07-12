"""Typed, deterministic evidence aggregation and entity resolution utilities."""

import re
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from backend.research.models import ResearchDocument
from backend.services.company.knowledge import KNOWLEDGE


class DomainResolver:
    """Normalize hosts and group corporate subdomains under registrable domains."""

    @staticmethod
    def normalize(value: str | None) -> str | None:
        if not value:
            return None
        candidate = value.strip().casefold()
        parsed = urlsplit(candidate if "://" in candidate else f"//{candidate}")
        return (parsed.hostname or "").removeprefix("www.").rstrip(".") or None

    @staticmethod
    def registrable(host: str) -> str:
        labels = host.casefold().removeprefix("www.").split(".")
        if len(labels) <= 2:
            return ".".join(labels)
        suffix = ".".join(labels[-2:])
        return ".".join(labels[-3:]) if suffix in KNOWLEDGE.two_level_suffixes else suffix

    @staticmethod
    def excluded(host: str) -> bool:
        return any(host == item or host.endswith(f".{item}") for item in KNOWLEDGE.excluded_domains)


class AliasResolver:
    """Normalize legal names, brands, aliases, and acronyms."""

    @staticmethod
    def normalize(value: str) -> str:
        tokens = re.findall(r"[a-z0-9]+", value.casefold())
        normalized = " ".join(token for token in tokens if token not in KNOWLEDGE.legal_suffixes)
        return KNOWLEDGE.aliases.get(normalized, normalized)

    @classmethod
    def equivalent(cls, left: str, right: str) -> bool:
        first, second = cls.normalize(left), cls.normalize(right)
        if not first or not second:
            return False
        if first == second:
            return True
        return first == "".join(word[0] for word in second.split()) or second == "".join(word[0] for word in first.split())


class OrganizationMatcher:
    """Layered organization matching with edit similarity as the final fallback."""

    @staticmethod
    def _edit_similarity(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        previous = list(range(len(right) + 1))
        for row, left_character in enumerate(left, 1):
            current = [row]
            for column, right_character in enumerate(right, 1):
                current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (left_character != right_character)))
            previous = current
        return 1 - previous[-1] / max(len(left), len(right))

    @classmethod
    def matches(cls, left: str, right: str) -> bool:
        if AliasResolver.equivalent(left, right):
            return True
        first, second = AliasResolver.normalize(left), AliasResolver.normalize(right)
        first_tokens, second_tokens = set(first.split()), set(second.split())
        union = first_tokens | second_tokens
        jaccard = len(first_tokens & second_tokens) / len(union) if union else 0
        if jaccard >= 0.8:
            return True
        if cls._edit_similarity(first, second) >= 0.9:
            return True
        return SequenceMatcher(None, first, second).ratio() >= 0.92


class OfficialSiteDetector:
    """Detect official-site evidence from source, hostname, metadata, and legal markers."""

    @staticmethod
    def score(document: ResearchDocument, name: str | None, root_domain: str) -> float:
        host = DomainResolver.normalize(str(document.url)) if document.url else None
        if host is None or DomainResolver.registrable(host) != root_domain:
            return 0.0
        text = f"{document.title or ''} {document.content}".casefold()
        markers = sum(marker in text for marker in KNOWLEDGE.official_indicators)
        name_match = bool(name and AliasResolver.normalize(name) in AliasResolver.normalize(text))
        source_confirmation = document.source == "official_website"
        evidence = (int(source_confirmation), int(name_match), min(markers, 3) / 3)
        return sum(evidence) / len(evidence)


class NormalizedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document: ResearchDocument
    normalized_url: str
    host: str
    root_domain: str
    normalized_content: str
    source_quality: float = Field(ge=0, le=1)
    freshness: float = Field(ge=0, le=1)
    search_rank: int = Field(ge=1)


class EvidenceNormalizer:
    """Convert provider documents into comparable evidence records."""

    @classmethod
    def normalize(cls, documents: list[ResearchDocument]) -> list[NormalizedEvidence]:
        records = []
        for rank, document in enumerate(documents, 1):
            host = DomainResolver.normalize(str(document.url)) if document.url else None
            if host is None or DomainResolver.excluded(host):
                continue
            records.append(NormalizedEvidence(
                document=document,
                normalized_url=str(document.url).rstrip("/").casefold(),
                host=host,
                root_domain=DomainResolver.registrable(host),
                normalized_content=re.sub(r"[^a-z0-9]+", "", document.content.casefold())[:1000],
                source_quality=cls._source_quality(document, host),
                freshness=cls._freshness(document),
                search_rank=rank,
            ))
        return records

    @staticmethod
    def _source_quality(document: ResearchDocument, host: str) -> float:
        text = f"{document.title or ''} {document.content}".casefold()
        source_type = document.source
        if source_type == "official_website": pass
        elif "investor" in text: source_type = "investor_relations"
        elif host.endswith(".gov") or host == "sec.gov": source_type = "government"
        elif "linkedin.com" in host: source_type = "linkedin"
        elif "wikipedia.org" in host: source_type = "wikipedia"
        elif "crunchbase.com" in host: source_type = "crunchbase"
        elif host in {"reuters.com", "bloomberg.com"}: source_type = "news"
        return KNOWLEDGE.source_quality.get(source_type, KNOWLEDGE.source_quality["web_search"])

    @staticmethod
    def _freshness(document: ResearchDocument) -> float:
        published = re.search(r"Published:\s*(\d{4})", document.content, re.IGNORECASE)
        if published:
            age = max(0, date.today().year - int(published.group(1)))
            return 1 / (1 + age)
        years = [int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", document.content)]
        if years:
            return 1 / (1 + max(0, date.today().year - max(years)))
        return 0.7 if document.source == "official_website" else 0.6


class EvidenceDeduplicator:
    @staticmethod
    def deduplicate(records: list[NormalizedEvidence]) -> list[NormalizedEvidence]:
        unique: list[NormalizedEvidence] = []
        for record in records:
            match = next((index for index, current in enumerate(unique) if current.normalized_url == record.normalized_url or current.normalized_content == record.normalized_content), None)
            if match is None: unique.append(record)
            elif record.source_quality > unique[match].source_quality: unique[match] = record
        return unique


@dataclass(frozen=True)
class EvidenceCluster:
    key: str
    evidence: tuple[NormalizedEvidence, ...]


class EvidenceClusterer:
    @staticmethod
    def cluster(records: list[NormalizedEvidence]) -> list[EvidenceCluster]:
        groups: dict[str, list[NormalizedEvidence]] = {}
        for record in records: groups.setdefault(record.root_domain, []).append(record)
        return [EvidenceCluster(key, tuple(value)) for key, value in groups.items()]


class CompanyCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    canonical_name: str | None = None
    aliases: tuple[str, ...] = ()
    official_domains: tuple[str, ...] = ()
    supporting_urls: tuple[HttpUrl, ...] = ()


class CandidateFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    official_source_count: int
    independent_domains: int
    official_confirmations: int
    source_quality: float
    agreement: float
    entity_consistency: float
    freshness: float
    conflicting_candidates: int
    search_rank: float
    evidence_count: int


class ConfidenceFactors(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    official_evidence: float
    domain_agreement: float
    entity_agreement: float
    source_diversity: float
    freshness: float
    conflict_factor: float
    evidence_coverage: float
    confidence: float


@dataclass(frozen=True)
class ResolvedCandidate:
    candidate: CompanyCandidate
    features: CandidateFeatures
    factors: ConfidenceFactors


class ConfidenceCalibrator:
    """Reproducibly derive confidence factors from candidate features."""
    @staticmethod
    def calibrate(features: CandidateFeatures) -> ConfidenceFactors:
        official = features.official_confirmations / (features.official_confirmations + 1)
        diversity = features.independent_domains / (features.independent_domains + 1)
        coverage = features.evidence_count / (features.evidence_count + 2)
        rank = 1 / max(1, features.search_rank)
        conflict = 1 / (1 + features.conflicting_candidates)
        factors = (official, features.agreement, features.entity_consistency, diversity, features.freshness, coverage, features.source_quality, rank)
        confidence = (sum(factors) / len(factors)) * conflict
        if confidence > 0.95 and features.official_confirmations < 2: confidence = 0.95
        return ConfidenceFactors(official_evidence=official, domain_agreement=features.agreement, entity_agreement=features.entity_consistency, source_diversity=diversity, freshness=features.freshness, conflict_factor=conflict, evidence_coverage=coverage, confidence=confidence)


class CandidateBuilder:
    @classmethod
    def build(cls, clusters: list[EvidenceCluster], name_hint: str | None, domain_hint: str | None) -> list[ResolvedCandidate]:
        results: list[ResolvedCandidate] = []
        normalized_hint = DomainResolver.normalize(domain_hint)
        for cluster in clusters:
            aliases = cls._aliases(cluster, name_hint)
            canonical = name_hint if name_hint and any(OrganizationMatcher.matches(name_hint, alias) for alias in aliases) else (aliases[0] if aliases else None)
            matches = sum(cls._matches(record.document, canonical) for record in cluster.evidence) if canonical else 0
            agreement = max(matches / len(cluster.evidence), float(bool(normalized_hint and DomainResolver.registrable(normalized_hint) == cluster.key)))
            official = sum(OfficialSiteDetector.score(record.document, canonical, cluster.key) >= 0.6 for record in cluster.evidence)
            candidate = CompanyCandidate(canonical_name=canonical, aliases=tuple(dict.fromkeys(aliases)), official_domains=tuple(dict.fromkeys(record.host for record in cluster.evidence)), supporting_urls=tuple(record.document.url for record in cluster.evidence if record.document.url))
            features = CandidateFeatures(official_source_count=sum(record.document.source == "official_website" for record in cluster.evidence), independent_domains=len({record.host for record in cluster.evidence}), official_confirmations=official, source_quality=sum(record.source_quality for record in cluster.evidence) / len(cluster.evidence), agreement=agreement, entity_consistency=agreement, freshness=sum(record.freshness for record in cluster.evidence) / len(cluster.evidence), conflicting_candidates=max(0, len(clusters) - 1), search_rank=sum(record.search_rank for record in cluster.evidence) / len(cluster.evidence), evidence_count=len(cluster.evidence))
            results.append(ResolvedCandidate(candidate, features, ConfidenceCalibrator.calibrate(features)))
        return results

    @staticmethod
    def _aliases(cluster: EvidenceCluster, hint: str | None) -> list[str]:
        aliases = [hint] if hint else []
        for record in cluster.evidence:
            title = record.document.title or ""
            candidate = re.split(r"\s+[|–—-]\s+|:\s+", title, maxsplit=1)[0].strip()
            if 1 <= len(candidate.split()) <= 8: aliases.append(candidate)
        return [alias for alias in dict.fromkeys(aliases) if alias]

    @staticmethod
    def _matches(document: ResearchDocument, name: str | None) -> bool:
        if not name: return False
        text = f"{document.title or ''} {document.content}"
        return AliasResolver.normalize(name) in AliasResolver.normalize(text)


class CandidateRanker:
    @staticmethod
    def rank(candidates: list[ResolvedCandidate]) -> list[ResolvedCandidate]:
        return sorted(candidates, key=lambda item: item.factors.confidence, reverse=True)
