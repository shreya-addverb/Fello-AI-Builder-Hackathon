"""FastAPI application entry point."""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.analyze import router as analyze_router
from backend.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Backend foundation for account intelligence and enrichment.",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )
    application.include_router(analyze_router)
    application.include_router(analyze_router, prefix="/api", include_in_schema=False)

    @application.get("/health", tags=["Health"])
    async def health() -> dict[str, str]:
        """Return the health status of the API."""
        return {"status": "healthy"}

    @application.get("/api/health", include_in_schema=False)
    async def api_health() -> dict[str, str]:
        return {"status": "healthy"}

    @application.get("/api/system", tags=["Health"])
    async def system_status() -> dict[str, object]:
        """Expose safe runtime configuration without secret values."""
        return {
            "status": "healthy",
            "version": settings.app_version,
            "model": settings.gemini_model,
            "providers": {
                "gemini": settings.gemini_api_key is not None,
                "tavily": settings.tavily_api_key is not None,
                "firecrawl": settings.firecrawl_api_key is not None,
                "ip_ownership": True,
            },
            "research_timeout_seconds": settings.research_timeout_seconds,
        }

    frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        application.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

        @application.get("/{path:path}", include_in_schema=False)
        async def frontend(path: str) -> FileResponse:
            candidate = frontend_dist / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(frontend_dist / "index.html")

    return application


app = create_app()
