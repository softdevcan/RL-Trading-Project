"""Main FastAPI application"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.routes import health, items, trading, hyperopt
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

settings = get_settings()


@asynccontextmanager  # (#32) replaces deprecated @app.on_event
async def lifespan(app: FastAPI):
    print(f"Starting {settings.API_TITLE} v{settings.API_VERSION}")
    print(f"Docs available at: http://{settings.HOST}:{settings.PORT}/docs")
    yield
    print(f"Shutting down {settings.API_TITLE}")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Include routers
app.include_router(health.router)
app.include_router(items.router)
app.include_router(trading.router, prefix="/api")
app.include_router(hyperopt.router, prefix="/api")

# Serve static files (for web UI)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve web UI
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web UI"""
    ui_path = "static/index.html"
    if os.path.exists(ui_path):
        with open(ui_path, 'r', encoding='utf-8') as f:
            return f.read()
    return """
    <html>
        <head><title>RL Trading System</title></head>
        <body>
            <h1>RL Trading System</h1>
            <p>Web UI not found. Please create static/index.html</p>
            <p><a href="/docs">API Documentation</a></p>
        </body>
    </html>
    """


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon to prevent 404 errors"""
    from fastapi.responses import FileResponse
    favicon_path = "static/favicon.ico"
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    # Return 204 No Content instead of 404
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools():
    """Chrome DevTools metadata endpoint"""
    from fastapi.responses import Response
    return Response(status_code=204)


