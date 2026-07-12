# Backend requirements audit

This audit compares the implementation with the supplied Fello AI Builder brief. It reflects the backend after the targeted hardening changes in this workspace.

## Met

- Modular, single-owner stages for identification, enrichment, technology, leadership, signals, persona, intent, summary, and recommendations.
- Evidence-first LLM prompts, strict structured-output validation, null/empty results when evidence is insufficient, and URL validation against collected documents.
- Company-name/domain and visitor entry points, including public-IP organization lookup.
- Visitor persona and intent reasoning based on observed behavioral signals.
- Gemini model discovery and model failover, structured provider failures, and continuation after a stage failure.
- Executive synthesis and grounded recommendations that may only cite keyed upstream facts.
- Request-scoped caching of duplicate search and crawl operations.
- Frontend-visible stage status, timing, provider/model use, cache/fallback events, and errors.

## Partially met

- Enrichment attempts the core profile plus ownership status, revenue, ticker, geographic footprint, and per-field confidence/evidence. Availability still depends on corroborating public evidence.
- Identification resolves canonical domains and rejects unsupported LLM decisions. Alias resolution is evidence-driven rather than backed by a dedicated corporate-identity dataset.
- Technology discovery now supports CRM, marketing, analytics, cloud, frontend, backend, hosting, security, databases, AI platforms, and developer tools. Detection remains limited by what public pages and search results expose.
- Leadership covers senior commercial and technical decision makers, but completeness depends on public current-role evidence.
- Signals cover the requested research themes through queries and reasoning, but the existing public signal taxonomy maps some themes to `other`; importance is available in the contract but is not yet generated.
- Independent research calls within stages can overlap only where explicitly implemented; the top-level stages remain ordered because later signal, summary, and recommendation reasoning consumes earlier results.
- Firecrawl failure is non-fatal, but there is no equivalent full-page crawler behind Tavily; Tavily search evidence continues to be used when crawling is unavailable.

## Not claimed

- Precise revenue, employee counts, ownership, or visitor-to-company attribution without corroborating public/provider evidence.
- Guaranteed discovery of private technology stacks or executives absent from trustworthy public sources.
- Cross-request persistent caching or a distributed rate-limit coordinator.

These limitations are intentional: returning unknown values is preferable to manufacturing a complete-looking report.
