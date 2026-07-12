"""Shared execution logging for pass-through pipeline services."""

import logging

from backend.models.context import AnalysisContext


def log_execution_started(
    service_name: str,
    provider_name: str | None = None,
) -> None:
    """Log the beginning of a service execution."""
    logger = logging.getLogger(f"backend.services.{service_name}")
    logger.info("Running %s...", service_name)
    if provider_name is not None:
        logger.info("Using provider %s for %s.", provider_name, service_name)


def log_execution_completed(
    service_name: str,
    succeeded: bool | None = None,
    duration_ms: int | None = None,
) -> None:
    """Log the completion and optional outcome of a service execution."""
    logger = logging.getLogger(f"backend.services.{service_name}")
    if succeeded is not None:
        logger.info("%s execution succeeded: %s.", service_name, succeeded)
    if duration_ms is None:
        logger.info("Completed %s.", service_name)
    else:
        logger.info("Completed %s in %d ms.", service_name, duration_ms)


def log_provider_failure(service_name: str, provider_name: str) -> None:
    """Log a provider failure without including request or visitor data."""
    logger = logging.getLogger(f"backend.services.{service_name}")
    logger.warning("Provider %s failed during %s.", provider_name, service_name)


async def log_passthrough_execution(
    context: AnalysisContext,
    service_name: str,
) -> AnalysisContext:
    """Log a service boundary and return its context unchanged."""
    log_execution_started(service_name)
    log_execution_completed(service_name)
    return context
