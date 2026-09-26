"""Encoder registry — get_encoder(name) factory.

Selects the right encoder based on config. Falls back to HashEncoder
when torch/model weights are unavailable.
"""

from __future__ import annotations

from ml.encoders import EncoderBase, HashEncoder

_registry: dict[str, type[EncoderBase]] = {}


def register_encoder(name: str):
    """Decorator to register an encoder class."""
    def decorator(cls: type[EncoderBase]):
        _registry[name] = cls
        return cls
    return decorator


def get_encoder(name: str, *, dim: int = 768, device: str = "cpu") -> EncoderBase:
    """Get an encoder by name. Falls back to HashEncoder if unavailable."""
    if name in _registry:
        try:
            return _registry[name](dim=dim, device=device)
        except Exception:
            pass
    # Always fall back to hash encoder
    return HashEncoder(dim=dim, name=f"{name}-fallback")


__all__ = ["get_encoder", "register_encoder"]
