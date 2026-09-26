"""ML warmup — pre-load models at startup.

Called from app/main.py lifespan. Lazy-imports torch so the API
boots without ML dependencies installed.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def warm() -> None:
    """Pre-load ML models into memory.

    Called once at application startup. Each component handles its own
    fallback if torch/models are unavailable.
    """
    log.info("ml_warmup_start")

    try:
        from ml.encoders.registry import get_encoder
        from app.core.config import settings

        # Warm up encoders
        encoder = get_encoder(settings.paper_encoder, dim=settings.embedding_dim)
        encoder.warmup()
        log.info("encoder_warmed: %s", encoder.name)
    except Exception as exc:
        log.warning("encoder_warmup_failed: %s", exc)

    try:
        from ml.ner import NERPipeline
        ner = NERPipeline()
        ner.warmup()
        log.info("ner_warmed")
    except Exception as exc:
        log.warning("ner_warmup_failed: %s", exc)

    try:
        from ml.relation import REPipeline
        re = REPipeline()
        re.warmup()
        log.info("re_warmed")
    except Exception as exc:
        log.warning("re_warmup_failed: %s", exc)

    try:
        from ml.gnn.model import CKGAI_GNN
        gnn = CKGAI_GNN()
        gnn.warmup()
        log.info("gnn_warmed")
    except Exception as exc:
        log.warning("gnn_warmup_failed: %s", exc)

    log.info("ml_warmup_complete")
