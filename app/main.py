from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import structlog
import time

from core.config import settings
from core.auth import verify_api_key_and_get_developer
from core.audit import get_audit_middleware
from db.session import init_db, get_db
from api.routes import health, predict_tabular, predict_image, explain
from models.tabular_inference import initialize_tabular_model
from models.imaging_inference import initialize_imaging_model
from datetime import datetime, timezone


# Structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"]
)

ACTIVE_REQUESTS = Gauge(
    "http_requests_active",
    "Active HTTP requests",
    ["method", "endpoint"]
)

INFERENCE_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["model_type", "status"]
)

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
    ["model_type"]
)


def get_rate_limit_key(request: Request) -> str:
    """Extract API key for rate limiting, fallback to IP."""
    return request.headers.get("X-API-Key", get_remote_address(request))


storage_uri = settings.REDIS_URL if settings.REDIS_ENABLED else "memory://"

limiter = Limiter(
    key_func=get_rate_limit_key,
    default_limits=[f"{settings.RATE_LIMIT_REQUESTS}/minute"],
    storage_uri=storage_uri,
)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_code": "VALIDATION_ERROR",
            "detail": [
                {"field": err["loc"][-1] if err["loc"] else "unknown", "issue": err["msg"]}
                for err in exc.errors()
            ],
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    )


async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    """Custom rate limit handler with retry_after_seconds."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "status": "error",
            "error_code": "RATE_LIMIT_EXCEEDED",
            "detail": f"Limit of {settings.RATE_LIMIT_REQUESTS} requests/minute exceeded for this API key.",
            "retry_after_seconds": exc.retry_after,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await init_db()
    await initialize_tabular_model()
    await initialize_imaging_model()
    logger.info("application_started")
    yield
    logger.info("application_shutdown")


app = FastAPI(
    title="Dil API",
    description="Multi-Modal Cardiovascular Risk Diagnostic Platform",
    version="1.0.0",
    lifespan=lifespan,
)


def get_rate_limit_key(request: Request) -> str:
    """Extract API key for rate limiting, fallback to IP."""
    return request.headers.get("X-API-Key", get_remote_address(request))


storage_uri = settings.REDIS_URL if settings.REDIS_ENABLED else "memory://"

limiter = Limiter(
    key_func=get_rate_limit_key,
    default_limits=[f"{settings.RATE_LIMIT_REQUESTS}/minute"],
    storage_uri=storage_uri,
)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_code": "VALIDATION_ERROR",
            "detail": [
                {"field": err["loc"][-1] if err["loc"] else "unknown", "issue": err["msg"]}
                for err in exc.errors()
            ],
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    )


async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    """Custom rate limit handler with retry_after_seconds."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "status": "error",
            "error_code": "RATE_LIMIT_EXCEEDED",
            "detail": f"Limit of {settings.RATE_LIMIT_REQUESTS} requests/minute exceeded for this API key.",
            "retry_after_seconds": exc.retry_after,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    )


# Prometheus metrics middleware
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path
    ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).inc()
    start_time = time.time()
    
    response: Response = await call_next(request)
    
    duration = time.time() - start_time
    ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).dec()
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=response.status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
    
    return response


# Audit logging middleware
from core.audit import get_audit_middleware
audit_middleware = get_audit_middleware()
app.add_middleware(audit_middleware, excluded_paths=["/healthz", "/docs", "/openapi.json", "/redoc", "/metrics"])

# CORS - restrict to configured origins
allowed_origins = [o.strip() for o in settings.CORS_ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
)

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Prometheus metrics middleware
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path
    ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).inc()
    start_time = time.time()
    
    response: Response = await call_next(request)
    
    duration = time.time() - start_time
    ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).dec()
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=response.status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
    
    return response


# Health check
@app.get("/healthz")
async def health_check():
    from api.routes.health import health_check as health_check_impl
    from db.session import get_db
    from sqlalchemy.ext.asyncio import AsyncSession
    
    async for db in get_db():
        return await health_check_impl(db)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await init_db()
    await initialize_tabular_model()
    await initialize_imaging_model()
    logger.info("application_started")
    yield
    logger.info("application_shutdown")


app = FastAPI(
    title="Dil API",
    description="Multi-Modal Cardiovascular Risk Diagnostic Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Health check
@app.get("/healthz")
async def health_check():
    from api.routes.health import health_check as health_check_impl
    from db.session import get_db
    from sqlalchemy.ext.asyncio import AsyncSession
    
    async for db in get_db():
        return await health_check_impl(db)

app.include_router(health.router)
app.include_router(predict_tabular.router)
app.include_router(predict_image.router)
app.include_router(explain.router)