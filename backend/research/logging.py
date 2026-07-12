"""Shared, privacy-safe logging for external research operations."""

import json
import logging
from collections.abc import Mapping


def log_provider_started(provider: str, operation: str) -> None:
    """Log provider invocation without request contents."""
    logging.getLogger("backend.research").info(
        "Provider %s started %s.", provider, operation
    )


def log_provider_completed(provider: str, operation: str, duration_ms: int) -> None:
    """Log successful provider completion and duration."""
    logging.getLogger("backend.research").info(
        "Provider %s completed %s in %d ms.", provider, operation, duration_ms
    )


def log_outbound_request(
    provider: str,
    method: str,
    url: str,
    timeout_seconds: float,
    headers: Mapping[str, str],
    payload: object,
) -> None:
    """Log safe outbound request metadata with authentication values redacted."""
    safe_headers = {
        name: "<redacted>" if name.lower() in {"authorization", "x-goog-api-key"} else value
        for name, value in headers.items()
    }
    payload_size = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    logging.getLogger("backend.research").info(
        "Provider %s outbound request method=%s url=%s timeout_seconds=%s "
        "headers=%s payload_bytes=%d.",
        provider,
        method,
        url,
        timeout_seconds,
        safe_headers,
        payload_size,
    )


def log_outbound_response(
    provider: str,
    status_code: int,
    headers: Mapping[str, str],
    body: str,
) -> None:
    """Log response diagnostics, truncating the body to a safe bounded excerpt."""
    logging.getLogger("backend.research").info(
        "Provider %s response status=%d headers=%s body_excerpt=%r.",
        provider,
        status_code,
        dict(headers),
        body[:500],
    )


def log_provider_failed(
    provider: str,
    operation: str,
    duration_ms: int,
    *,
    exception_type: str | None = None,
    error_message: str | None = None,
) -> None:
    """Log provider failure details without authentication secrets."""
    logging.getLogger("backend.research").warning(
        "Provider %s failed %s after %d ms exception_type=%s error=%r.",
        provider,
        operation,
        duration_ms,
        exception_type,
        error_message,
    )
