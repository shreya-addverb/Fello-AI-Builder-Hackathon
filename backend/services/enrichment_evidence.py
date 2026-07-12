"""Deterministic evidence processing for company enrichment."""

import re
from datetime import date
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from backend.research.models import ResearchDocument


SOURCE_QUALITY = {
    "official_website": 1.0,
    "official_about": 0.98,
    "official_company": 0.98,
    "investor_relations": 0.96,
    "sec": 0.95,
    "government": 0.92,
    "press_release": 0.86,
    "wikipedia": 0.58,
    "linkedin": 0.48,
    "news": 0.68,
    "web_search": 0.6,
}


class NormalizedEnrichmentEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document: ResearchDocument
    normalized_url: str
    normalized_content: str
    source_quality: float = Field(ge=0, le=1)
    freshness: float = Field(ge=0, le=1)


class FieldCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    value: str
    supporting_urls: tuple[HttpUrl, ...]
    source_quality: float = Field(ge=0, le=1)
    freshness: float = Field(ge=0, le=1)
    agreement: float = Field(ge=0, le=1)
    confirmations: int = Field(ge=1)
    official_confirmations: int = Field(ge=0)
    conflict_penalty: float = Field(ge=0, le=1)
    score: float = Field(ge=0, le=1)


class DocumentNormalizer:
    """Canonicalize URLs/content and attach deterministic quality and freshness."""

    @classmethod
    def normalize(cls, documents: list[ResearchDocument]) -> list[NormalizedEnrichmentEvidence]:
        records = [cls._normalize(document) for document in documents if document.url]
        return cls._deduplicate(records)

    @classmethod
    def _normalize(cls, document: ResearchDocument) -> NormalizedEnrichmentEvidence:
        parsed = urlsplit(str(document.url))
        host = (parsed.hostname or "").casefold().removeprefix("www.")
        path = parsed.path.rstrip("/") or "/"
        normalized_url = urlunsplit(("https", host, path, "", ""))
        content = re.sub(r"\s+", " ", f"{document.title or ''} {document.content}").strip()
        return NormalizedEnrichmentEvidence(
            document=document,
            normalized_url=normalized_url,
            normalized_content=content,
            source_quality=cls._source_quality(document, host, content),
            freshness=cls._freshness(content, document.source),
        )

    @staticmethod
    def _source_quality(document: ResearchDocument, host: str, content: str) -> float:
        if document.source in SOURCE_QUALITY:
            return SOURCE_QUALITY[document.source]
        if host.endswith(".gov") or host == "sec.gov":
            return SOURCE_QUALITY["government"]
        if "investor" in content.casefold():
            return SOURCE_QUALITY["investor_relations"]
        if "wikipedia.org" in host:
            return SOURCE_QUALITY["wikipedia"]
        if "linkedin.com" in host:
            return SOURCE_QUALITY["linkedin"]
        return SOURCE_QUALITY["web_search"]

    @staticmethod
    def _freshness(content: str, source: str) -> float:
        published = re.search(r"Published:\s*(\d{4})", content, re.IGNORECASE)
        years = [int(published.group(1))] if published else [
            int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", content)
        ]
        if not years:
            return 0.8 if source.startswith("official_") else 0.6
        return 1 / (1 + max(0, date.today().year - max(years)))

    @staticmethod
    def _deduplicate(records: list[NormalizedEnrichmentEvidence]) -> list[NormalizedEnrichmentEvidence]:
        unique: list[NormalizedEnrichmentEvidence] = []
        for record in records:
            fingerprint = re.sub(r"[^a-z0-9]", "", record.normalized_content.casefold())[:600]
            duplicate = next((index for index, current in enumerate(unique) if current.normalized_url == record.normalized_url or re.sub(r"[^a-z0-9]", "", current.normalized_content.casefold())[:600] == fingerprint), None)
            if duplicate is None:
                unique.append(record)
            elif record.source_quality > unique[duplicate].source_quality:
                unique[duplicate] = record
        return unique


class EnrichmentEvidenceAggregator:
    """Extract, normalize, group, score, and explain field candidates."""

    _PATTERNS = {
        "founded_year": (r"\b(?:founded|established|incorporated)\s+(?:in\s+)?((?:18|19|20)\d{2})\b",),
        "stock_ticker": (r"\b(?:NYSE|NASDAQ|LSE|NSE|BSE)\s*[:\uFF1A]\s*([A-Z][A-Z0-9.-]{0,9})\b",),
        "headquarters": (r"\b(?:headquarters|headquartered|based)\s+(?:is\s+)?(?:in|at)\s+([^.;\n]{3,80})",),
        "employee_count": (r"\b([\d,.]+\s*[kKmM]?\+?)\s+employees\b",),
        "revenue": (r"\b(?:revenue|annual revenue)\s*(?:of|was|is|:)\s*([^.;\n]{2,40})",),
        "industry": (r"\b(?:industry|sector)\s*(?:is|:)?\s*([^.;\n]{3,80})",),
    }

    @classmethod
    def aggregate(
        cls,
        records: list[NormalizedEnrichmentEvidence],
        identified_name: str | None,
        identified_domain: str | None,
    ) -> tuple[list[FieldCandidate], dict[str, tuple[str, ...]]]:
        raw: dict[str, list[tuple[str, NormalizedEnrichmentEvidence]]] = {}
        if identified_name:
            for record in records:
                if cls._contains(record.normalized_content, identified_name):
                    raw.setdefault("canonical_company_name", []).append((identified_name, record))
        for record in records:
            host = (urlsplit(record.normalized_url).hostname or "").removeprefix("www.")
            if identified_domain and (host == identified_domain or host.endswith(f".{identified_domain}")):
                raw.setdefault("website", []).append((f"https://{identified_domain}/", record))
            cls._extract_patterns(record, raw)
            cls._extract_ownership(record, raw)

        conflicts = {
            field: tuple(values)
            for field, entries in raw.items()
            if len(values := list(dict.fromkeys(value for value, _ in entries))) > 1
        }
        candidates = [
            cls._candidate(field, value, entries, conflicts)
            for field, field_entries in raw.items()
            for value in dict.fromkeys(item[0] for item in field_entries)
            if (entries := [item for item in field_entries if item[0] == value])
        ]
        return sorted(candidates, key=lambda item: item.score, reverse=True), conflicts

    @classmethod
    def _extract_patterns(cls, record: NormalizedEnrichmentEvidence, raw: dict[str, list[tuple[str, NormalizedEnrichmentEvidence]]]) -> None:
        for field, patterns in cls._PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, record.normalized_content, re.IGNORECASE):
                    value = cls._normalize_value(field, match.group(1).strip(" ,"))
                    if value:
                        raw.setdefault(field, []).append((value, record))

    @staticmethod
    def _extract_ownership(record: NormalizedEnrichmentEvidence, raw: dict[str, list[tuple[str, NormalizedEnrichmentEvidence]]]) -> None:
        text = record.normalized_content.casefold()
        ownership = "public" if any(marker in text for marker in ("public company", "nyse:", "nasdaq:", "listed company")) else "private" if "privately held" in text else "government" if "government-owned" in text else "nonprofit" if "nonprofit" in text or "non-profit" in text else None
        if ownership:
            raw.setdefault("ownership_type", []).append((ownership, record))

    @staticmethod
    def _normalize_value(field: str, value: str) -> str:
        if field == "employee_count":
            match = re.fullmatch(r"([\d,.]+)\s*([kKmM]?)\+?", value)
            if not match:
                return ""
            number = float(match.group(1).replace(",", ""))
            multiplier = {"k": 1_000, "m": 1_000_000}.get(match.group(2).casefold(), 1)
            return str(int(number * multiplier))
        if field == "stock_ticker":
            return value.upper()
        return re.sub(r"\s+", " ", value).strip()

    @classmethod
    def _candidate(cls, field: str, value: str, entries: list[tuple[str, NormalizedEnrichmentEvidence]], conflicts: dict[str, tuple[str, ...]]) -> FieldCandidate:
        records = [record for _, record in entries]
        confirmations = len({
            urlsplit(record.normalized_url).hostname for record in records
        })
        official = len({
            urlsplit(record.normalized_url).hostname
            for record in records
            if record.document.source.startswith("official_")
            or record.document.source == "investor_relations"
        })
        quality = sum(record.source_quality for record in records) / len(records)
        freshness = sum(record.freshness for record in records) / len(records)
        field_values = conflicts.get(field, (value,))
        agreement = confirmations / (confirmations + max(0, len(field_values) - 1))
        conflict_penalty = 1 / max(1, len(field_values))
        score = ((quality + freshness + agreement + confirmations / (confirmations + 1) + official / (official + 1)) / 5) * conflict_penalty
        urls = tuple(dict.fromkeys(
            record.document.url for record in records if record.document.url
        ))
        return FieldCandidate(field=field, value=value, supporting_urls=urls, source_quality=quality, freshness=freshness, agreement=agreement, confirmations=confirmations, official_confirmations=official, conflict_penalty=conflict_penalty, score=score)

    @staticmethod
    def _contains(text: str, value: str) -> bool:
        normalized_text = re.sub(r"[^a-z0-9]", "", text.casefold())
        normalized_value = re.sub(r"[^a-z0-9]", "", value.casefold())
        return bool(normalized_value and normalized_value in normalized_text)


class EvidenceValidator:
    """Verify every populated output field against candidates or verbatim evidence."""

    @classmethod
    def supported(cls, field: str, value: object, candidates: list[FieldCandidate], records: list[NormalizedEnrichmentEvidence]) -> bool:
        if isinstance(value, list):
            return bool(value) and all(
                any(cls._contains(record.normalized_content, str(item)) for record in records)
                for item in value
            )
        serialized = str(value)
        if any(candidate.field == field and cls._equivalent(field, serialized, candidate.value) and candidate.score >= 0.35 for candidate in candidates):
            return True
        return field in {"business_category", "business_description", "geographic_footprint", "company_size"} and any(cls._contains(record.normalized_content, serialized) for record in records)

    @staticmethod
    def _equivalent(field: str, left: str, right: str) -> bool:
        if field in {"employee_count", "founded_year"}:
            return re.sub(r"\D", "", left) == re.sub(r"\D", "", right)
        if field == "website":
            return (urlsplit(left).hostname or "").removeprefix("www.") == (urlsplit(right).hostname or "").removeprefix("www.")
        return re.sub(r"[^a-z0-9]", "", left.casefold()) == re.sub(r"[^a-z0-9]", "", right.casefold())

    @staticmethod
    def _contains(text: str, value: str) -> bool:
        return re.sub(r"[^a-z0-9]", "", value.casefold()) in re.sub(r"[^a-z0-9]", "", text.casefold())
