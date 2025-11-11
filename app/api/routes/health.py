"""Health check endpoints"""

from fastapi import APIRouter

from app.schemas.health import HealthResponse
from app import __version__

router = APIRouter(tags=["Health"])


@router.get("/", response_model=dict)
async def root():
    """Root endpoint - welcome message"""
    return {
        "message": "Welcome to RL Trading API",
        "version": __version__,
        "docs": "/docs"
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="API is running",
        version=__version__
    )
