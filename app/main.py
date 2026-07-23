"""Main FastAPI application"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.config import get_settings
from app.api.routes import health, items, trading, hyperopt, prediction, config
from fastapi.staticfiles import StaticFiles
import os

settings = get_settings()


@asynccontextmanager  # (#32) replaces deprecated @app.on_event
async def lifespan(app: FastAPI):
    print(f"Starting {settings.API_TITLE} v{settings.API_VERSION}")
    print(f"Dashboard:  http://{settings.HOST}:{settings.PORT}/dash/")
    print(f"API docs:   http://{settings.HOST}:{settings.PORT}/docs")

    if settings.AUTH_ENABLED:
        # Auth semasi + ilk admin. JWT anahtari eksikse burada fail-fast olur.
        from app.auth.db import init_db, session_scope
        from app.auth.security import get_secret_key
        from app.auth.service import purge_expired_sessions
        from app.auth.bootstrap import bootstrap_admin

        get_secret_key()
        init_db()
        with session_scope() as db:
            purge_expired_sessions(db)
        bootstrap_admin()
        print("Auth:       ACIK — giris: /login")
    else:
        print("Auth:       KAPALI (AUTH_ENABLED=False) — sunucuda kullanmayin!")

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

# Oturum kapisi. CORS'tan ONCE eklenir; Starlette middleware'i ters sirada
# uyguladigi icin CORS en distaki katman olur ve 401 yanitlari ile preflight
# istekleri de dogru CORS basliklarini alir.
from app.auth.middleware import AuthGateMiddleware
app.add_middleware(AuthGateMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Include routers
from app.auth.routes import router as auth_router
from app.api.routes import admin as admin_routes

app.include_router(auth_router)                      # /login, /auth/*
app.include_router(admin_routes.router, prefix="/api")  # /api/admin/*
app.include_router(health.router)
app.include_router(items.router)
app.include_router(trading.router, prefix="/api")
app.include_router(hyperopt.router, prefix="/api")
app.include_router(prediction.router, prefix="/api")
app.include_router(config.router, prefix="/api")

# Serve static files (legacy UI kept as reference)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Dash frontend ──────────────────────────────────────────────────────────
from starlette.middleware.wsgi import WSGIMiddleware
from dashboard.app import create_dash_app, PrefixMiddleware

_DASH_PREFIX = "/dash/"
_dash_app = create_dash_app(_DASH_PREFIX)
_prefix_mw = PrefixMiddleware(_dash_app.server, "/dash")
app.mount("/dash", WSGIMiddleware(_prefix_mw))

# ── Root redirect ──────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Redirect root to Dash dashboard."""
    return RedirectResponse(url="/dash/")


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon to prevent 404 errors."""
    from fastapi.responses import FileResponse, Response
    favicon_path = "static/favicon.ico"
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return Response(status_code=204)


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools():
    """Chrome DevTools metadata endpoint."""
    from fastapi.responses import Response
    return Response(status_code=204)
