"""
FastAPI Application Entry Point
Main FastAPI application with routers for all API endpoints
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import alerts, dga, health, predictions, reports, transformers


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="TransformerGuard API",
        description="AI-powered transformer health scoring and failure prediction",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(
        transformers.router, prefix="/api/v1/transformers", tags=["Transformers"]
    )
    app.include_router(dga.router, prefix="/api/v1/dga", tags=["DGA"])
    app.include_router(health.router, prefix="/api/v1/health", tags=["Health Index"])
    app.include_router(
        predictions.router, prefix="/api/v1/predictions", tags=["Predictions"]
    )
    app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["Alerts"])
    app.include_router(reports.router, prefix="/api/v1/reports", tags=["Reports"])

    @app.get("/health")
    async def health_check():
        """Health check endpoint for monitoring."""
        return {"status": "healthy", "service": "TransformerGuard"}

    return app


# Create the FastAPI application instance
app = create_app()
