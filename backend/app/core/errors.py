"""Error hierarchy and FastAPI exception handlers.

Every failure the client sees is `{detail, code?, requestId}` — never a stack trace.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.core.logging import get_logger, request_id_var

log = get_logger(__name__)


class AppError(Exception):
    """Base for all errors we raise deliberately."""

    status_code: int = status.HTTP_400_BAD_REQUEST
    code: str = "bad_request"

    def __init__(self, detail: str, *, code: str | None = None, **extra: Any) -> None:
        super().__init__(detail)
        self.detail = detail
        if code:
            self.code = code
        self.extra = extra

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"detail": self.detail, "code": self.code}
        payload.update(self.extra)
        if rid := request_id_var.get():
            payload["requestId"] = rid
        return payload


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ConflictError(AppError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"


class ValidationAppError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "validation_error"


class UnauthorizedError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"


class ForbiddenError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "forbidden"


class RateLimitError(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"


class ServiceUnavailableError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "service_unavailable"


def _envelope(status_code: int, detail: str, code: str, exc: Exception | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"detail": detail, "code": code}
    if rid := request_id_var.get():
        payload["requestId"] = rid
    return payload


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            log.error("app_error", detail=exc.detail, code=exc.code, **exc.extra)
        else:
            log.warning("app_error", detail=exc.detail, code=exc.code, **exc.extra)
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload())

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [
            {"field": ".".join(str(p) for p in e.get("loc", ())), "message": e.get("msg", "")}
            for e in exc.errors()
        ]
        payload = _envelope(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Request validation failed",
            "validation_error",
        )
        payload["fields"] = fields
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload)

    @app.exception_handler(IntegrityError)
    async def _integrity(_: Request, exc: IntegrityError) -> JSONResponse:
        log.warning("integrity_error", error=str(exc.orig))
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content=_envelope(status.HTTP_409_CONFLICT, "Constraint violated", "conflict"),
        )

    @app.exception_handler(SQLAlchemyError)
    async def _sqlalchemy(_: Request, exc: SQLAlchemyError) -> JSONResponse:
        log.error("database_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_envelope(
                status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable", "service_unavailable"
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception(
            "unhandled_exception",
            path=str(request.url.path),
            method=request.method,
            error=repr(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope(
                status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error", "internal_error"
            ),
        )
