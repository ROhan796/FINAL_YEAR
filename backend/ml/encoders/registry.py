"""Encoder registry — get_encoder(name) factory.

Selects the right encoder based on config. Falls back to HashEncoder
when torch/model weights are unavailable.
"""

from __future__ import annotations

from ml.encoders import EncoderBase, HashEncoder

_registry: dict[str, type[EncoderBase]] = {}
_loaded = False


def register_encoder(name: str):
    """Decorator to register an encoder class."""
    def decorator(cls: type[EncoderBase]):
        _registry[name] = cls
        return cls
    return decorator


def _ensure_loaded() -> None:
    """Import encoder modules so their @register_encoder decorators run."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for mod in ("ml.encoders.scibert", "ml.encoders.specter2"):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001 - optional deps may be missing
            pass


def resolve_device(device: str | None) -> str:
    """Map settings' 'auto' to a concrete torch device string."""
    if not device or device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"
    return device


def get_encoder(name: str, *, dim: int = 768, device: str = "cpu") -> EncoderBase:
    """Get an encoder by name. Falls back to HashEncoder if unavailable."""
    _ensure_loaded()
    device = resolve_device(device)
    if name in _registry:
        try:
            return _registry[name](dim=dim, device=device)
        except Exception:
            pass
    # Always fall back to hash encoder
    return HashEncoder(dim=dim, name=f"{name}-fallback")


__all__ = ["get_encoder", "register_encoder", "resolve_device"]
