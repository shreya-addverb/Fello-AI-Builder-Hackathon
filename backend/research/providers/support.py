"""Shared provider timing and failure classification."""

from time import perf_counter

import httpx
from pydantic import ValidationError

from backend.research.models import ResearchFailure


def elapsed_milliseconds(started_at: float) -> int:
    """Return nonnegative elapsed milliseconds for an operation."""
    return max(0, round((perf_counter() - started_at) * 1000))


def missing_configuration_failure(provider: str) -> ResearchFailure:
    """Return a safe failure for incomplete provider configuration."""
    return ResearchFailure(
        provider=provider,
        code="missing_configuration",
        message="Provider configuration is incomplete.",
    )


def http_status_failure(
    provider: str, status_code: int, response_body: str
) -> ResearchFailure:
    """Classify an unsuccessful provider HTTP status."""
    if status_code == 429:
        return ResearchFailure(
            provider=provider,
            code="rate_limited",
            message="Provider rate limit was reached.",
            retryable=True,
            status_code=status_code,
            exception_type="httpx.HTTPStatusError",
            response_body_excerpt=response_body[:500],
        )
    return ResearchFailure(
        provider=provider,
        code="provider_error",
        message="Provider returned an unsuccessful response.",
        retryable=status_code >= 500,
        status_code=status_code,
        exception_type="httpx.HTTPStatusError",
        response_body_excerpt=response_body[:500],
    )


def exception_failure(provider: str, error: Exception) -> ResearchFailure:
    """Map expected transport and validation exceptions to safe failures."""
    if isinstance(error, httpx.TimeoutException):
        return ResearchFailure(
            provider=provider,
            code="timeout",
            message="Provider request timed out.",
            retryable=True,
            exception_type=qualified_exception_name(error),
            exception_message=str(error),
        )
    if isinstance(error, (ValueError, ValidationError)):
        return ResearchFailure(
            provider=provider,
            code="invalid_response",
            message="Provider returned an invalid response.",
            exception_type=qualified_exception_name(error),
            exception_message=str(error),
        )
    return ResearchFailure(
        provider=provider,
        code="provider_error",
        message="Provider request failed.",
        retryable=isinstance(error, httpx.RequestError),
        exception_type=qualified_exception_name(error),
        exception_message=str(error),
    )


def qualified_exception_name(error: BaseException) -> str:
    """Return the fully qualified concrete exception class name."""
    error_type = type(error)
    return f"{error_type.__module__}.{error_type.__name__}"
