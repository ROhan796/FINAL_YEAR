"""CKGAI ML configuration.

All hyperparameters from the CKGAI specification appendix. Values are read from
environment variables via the backend's Settings, but this module provides
ML-specific defaults that are independent of the FastAPI config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class EncoderConfig:
    """Encoder model configuration."""
    dim: int = 768
    paper_encoder: str = "specter2"
    passage_encoder: str = "scibert"
    query_encoder: str = "specter2"
    max_length: int = 512
    device: str = "cpu"
    batch_size: int = 32


@dataclass(frozen=True)
class NERConfig:
    """Named Entity Recognition configuration."""
    model_name: str = "allenai/scibert_scivocab_uncased"
    num_labels: int = 17  # 8 entity types * 2 (B/I) + 1 (O)
    max_length: int = 512
    dropout: float = 0.1
    use_crf: bool = True
    device: str = "cpu"
    batch_size: int = 16

    ENTITY_TYPES: tuple[str, ...] = (
        "METHOD", "MATERIAL", "DISEASE", "ALGORITHM",
        "DATASET", "PHENOMENON", "METRIC", "CHEMICAL",
    )

    @property
    def tag_names(self) -> list[str]:
        tags = ["O"]
        for et in self.ENTITY_TYPES:
            tags.extend([f"B-{et}", f"I-{et}"])
        return tags


@dataclass(frozen=True)
class REConfig:
    """Relation Extraction configuration."""
    model_name: str = "allenai/scibert_scivocab_uncased"
    num_relations: int = 9  # 8 relations + no_relation
    hidden_dim: int = 768
    max_length: int = 512
    dropout: float = 0.1
    device: str = "cpu"
    batch_size: int = 16

    RELATIONS: tuple[str, ...] = (
        "used_in", "derived_from", "applied_to", "improves",
        "treats", "causes", "measures", "contains",
    )

    @property
    def relation_labels(self) -> list[str]:
        return list(self.RELATIONS) + ["no_relation"]


@dataclass(frozen=True)
class GNNConfig:
    """Graph Neural Network configuration (spec appendix)."""
    num_layers: int = 3
    hidden_dim: int = 256
    rgcn_bases: int = 30
    gat_heads: int = 8
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    negative_samples: int = 5
    margin: float = 1.0
    max_grad_norm: float = 1.0
    num_epochs: int = 100
    early_stopping_patience: int = 10
    virtual_node: bool = True  # optional improvement from architecture doc
    device: str = "cpu"


@dataclass(frozen=True)
class DiscoveryConfig:
    """Discovery / link prediction configuration."""
    scorer: str = "distmult"  # distmult | transe | dot | mlp
    cross_domain_boost: float = 1.3
    top_k: int = 100
    candidate_top_k: int = 200  # ANN pre-filter per node
    entity_resolution_threshold: float = 0.92


@dataclass(frozen=True)
class GraphConfig:
    """Knowledge graph configuration."""
    neo4j_enabled: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    projector_batch_size: int = 1000


@dataclass(frozen=True)
class MLConfig:
    """Top-level ML configuration."""
    enabled: bool = True
    device: str = "cpu"
    checkpoint_dir: str = "./models/checkpoints"

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    ner: NERConfig = field(default_factory=NERConfig)
    re: REConfig = field(default_factory=REConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)


def get_ml_config() -> MLConfig:
    """Build ML config from environment / backend settings."""
    try:
        from app.core.config import settings
        return MLConfig(
            enabled=settings.ml_enabled,
            device=settings.ml_device,
            checkpoint_dir=settings.ml_checkpoint_dir,
            encoder=EncoderConfig(
                dim=settings.embedding_dim,
                paper_encoder=settings.paper_encoder,
                passage_encoder=settings.passage_encoder,
                device=settings.ml_device,
            ),
            ner=NERConfig(
                model_name=settings.ml_ner_model,
                device=settings.ml_device,
            ),
            re=REConfig(
                model_name=settings.ml_relation_model,
                device=settings.ml_device,
            ),
            gnn=GNNConfig(
                num_layers=settings.gnn_layers,
                hidden_dim=settings.gnn_hidden_dim,
                rgcn_bases=settings.gnn_rgc_bases,
                gat_heads=settings.gnn_gat_heads,
                dropout=settings.gnn_dropout,
                learning_rate=settings.gnn_learning_rate,
                batch_size=settings.gnn_batch_size,
                negative_samples=settings.gnn_negative_samples,
                margin=settings.gnn_margin,
                device=settings.ml_device,
            ),
            discovery=DiscoveryConfig(
                scorer=settings.discovery_scorer,
                cross_domain_boost=settings.discovery_cross_domain_boost,
                top_k=settings.discovery_top_k,
                entity_resolution_threshold=settings.ml_entity_resolution_threshold,
            ),
            graph=GraphConfig(
                neo4j_enabled=settings.neo4j_enabled,
                neo4j_uri=settings.neo4j_uri,
                neo4j_user=settings.neo4j_user,
                neo4j_password=settings.neo4j_password,
                neo4j_database=settings.neo4j_database,
                projector_batch_size=settings.graph_projector_batch,
            ),
        )
    except Exception:
        return MLConfig()


__all__ = ["MLConfig", "EncoderConfig", "NERConfig", "REConfig", "GNNConfig", "DiscoveryConfig", "GraphConfig", "get_ml_config"]
