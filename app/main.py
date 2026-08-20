from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from core.config import settings
from core.auth import verify_api_key_and_get_developer
from db.session import init_db, get_db
from api.routes import health, predict_tabular, predict_image
from models.tabular_inference import initialize_tabular_model
from models.imaging_inference import initialize_imaging_model
from datetime import datetime, timezone


limiter = Limiter(
    key_func=lambda request: request.headers.get("X-API-Key", get_remote_address(request)),
    default_limits=[f"{settings.RATE_LIMIT_REQUESTS}/minute"],
    storage_uri="memory://",
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await init_db()
    await initialize_tabular_model()
    await initialize_imaging_model()
    yield


app = FastAPI(
    title="Dil API",
    description="Multi-Modal Cardiovascular Risk Diagnostic Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict_tabular.router)
app.include_router(predict_image.router)