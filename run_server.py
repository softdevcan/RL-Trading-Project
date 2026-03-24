"""
Run the FastAPI server
"""

import uvicorn
from app.core.config import get_settings

if __name__ == "__main__":
    settings = get_settings()

    print("=" * 60)
    print("RL TRADING SYSTEM - FASTAPI SERVER")
    print("=" * 60)
    print(f"Server starting at: http://{settings.HOST}:{settings.PORT}")
    print(f"API Docs: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"Web UI: http://{settings.HOST}:{settings.PORT}/")
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,  # (#37) configurable via DEBUG env var
        log_level="info"
    )
