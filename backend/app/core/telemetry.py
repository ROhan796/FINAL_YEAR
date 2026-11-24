"""OpenTelemetry instrumentation — enabled via ``OTEL_ENABLED=true``.

Degrades cleanly to a no-op when opentelemetry packages are absent or disabled,
so local development and tests carry zero overhead.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_configured = False


def configure_telemetry(app: Any) -> bool:
    """Attach OpenTelemetry tracer + FastAPI instrumentation when enabled."""
    global _configured
    if _configured or not settings.otel_enabled:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        resource = Resource.create({"service.name": "ckgai-api", "deployment.environment": settings.app_env})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        _configured = True
        log.info("opentelemetry_configured", env=settings.app_env)
        return True
    except ImportError as exc:
        log.warning("opentelemetry_packages_missing", error=str(exc))
        return False
    except Exception as exc:  # noqa: BLE001
        log.warning("opentelemetry_init_failed", error=str(exc))
        return False


def get_tracer(name: str = "ckgai"):
    """Return an OpenTelemetry tracer or a lightweight no-op context manager."""
    if not settings.otel_enabled:
        return _NoopTracer()
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except Exception:  # noqa: BLE001
        return _NoopTracer()


class _NoopSpan:
    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def set_attribute(self, key: str, value: object) -> None:  # noqa: ARG002
        return None


class _NoopTracer:
    def start_as_current_span(self, name: str, **_kwargs: object) -> _NoopSpan:  # noqa: ARG002
        return _NoopSpan()


__all__ = ["configure_telemetry", "get_tracer"]
