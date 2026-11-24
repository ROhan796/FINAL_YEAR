"""Relation Extraction training on scientific entity-pair examples.

Trains the pair MLP classifier `[h_e1 || h_e2 || h_e1*h_e2 || h_cls]` -> 9 classes
and saves checkpoint `models/checkpoints/re_head.pt`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ml.relation.types import NUM_RELATIONS, REL2ID, RELATION_LABELS

log = logging.getLogger(__name__)

_RE_SAMPLES: list[tuple[str, str, str, str]] = [
    ("Graph Attention Networks improve node classification accuracy on Cora.", "Graph Attention Networks", "accuracy", "improves"),
    ("Transformer is used in modern language modeling and neural machine translation.", "Transformer", "language modeling", "used_in"),
    ("BERT is derived from the Transformer encoder architecture.", "BERT", "Transformer", "derived_from"),
    ("CRISPR-Cas9 treats sickle cell disease by editing BCL11A.", "CRISPR-Cas9", "sickle cell disease", "treats"),
    ("Graphene is used in high-sensitivity biosensor platforms.", "Graphene", "biosensor", "used_in"),
    ("Beta-amyloid aggregation causes neurodegenerative disease progression.", "Beta-amyloid", "neurodegenerative disease", "causes"),
    ("AUC-ROC measures link prediction quality in biomedical knowledge graphs.", "AUC-ROC", "link prediction", "measures"),
    ("Lipid nanoparticle contains mRNA cargo for targeted delivery.", "Lipid nanoparticle", "mRNA", "contains"),
    ("R-GCN applied to molecular graphs predicts drug toxicity.", "R-GCN", "drug toxicity", "applied_to"),
    ("RoBERTa is derived from BERT with dynamic masking.", "RoBERTa", "BERT", "derived_from"),
    ("Contrastive learning improves retrieval recall across scientific corpora.", "Contrastive learning", "recall", "improves"),
    ("Doxorubicin treats glioblastoma in combination with nanocarriers.", "Doxorubicin", "glioblastoma", "treats"),
    ("Dataset ImageNet and optimizer Adam are mentioned separately in Section 2.", "ImageNet", "Adam", "no_relation"),
    ("We run experiments on PubMed and report wall-clock latency.", "PubMed", "latency", "no_relation"),
]


def _span_embedding(tokenizer, backbone, sentence: str, span_text: str):
    import torch

    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        hidden = backbone(**enc).last_hidden_state[0]  # (seq, 768)
    cls = hidden[0]
    span_tokens = set(tokenizer.tokenize(span_text.lower()))
    seq_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    idxs = [i for i, tok in enumerate(seq_tokens) if tok.lstrip("#").lower() in span_tokens]
    if idxs:
        return hidden[idxs].mean(dim=0), cls
    return hidden[1:-1].mean(dim=0) if hidden.size(0) > 2 else cls, cls


def train_re_checkpoint(
    output_path: Path,
    *,
    model_name: str = "allenai/scibert_scivocab_uncased",
    epochs: int = 40,
) -> dict[str, float]:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    hdim = int(backbone.config.hidden_size)
    pair_dim = hdim * 4

    clf = nn.Sequential(
        nn.Linear(pair_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(256, NUM_RELATIONS),
    )

    xs = []
    ys = []
    for sent, e1, e2, rel in _RE_SAMPLES:
        h1, cls = _span_embedding(tokenizer, backbone, sent, e1)
        h2, _ = _span_embedding(tokenizer, backbone, sent, e2)
        xs.append(torch.cat([h1, h2, h1 * h2, cls], dim=-1))
        ys.append(REL2ID[rel])

    X = torch.stack(xs)
    Y = torch.tensor(ys, dtype=torch.long)

    opt = torch.optim.AdamW(clf.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    clf.train()
    final_loss = 0.0
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(clf(X), Y)
        loss.backward()
        opt.step()
        final_loss = float(loss.item())

    clf.eval()
    with torch.no_grad():
        acc = float((clf(X).argmax(dim=-1) == Y).float().mean().item())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "pair_dim": pair_dim,
            "num_relations": NUM_RELATIONS,
            "relation_labels": RELATION_LABELS,
            "state_dict": clf.state_dict(),
            "metrics": {"loss": final_loss, "train_acc": acc},
        },
        output_path,
    )
    log.info("re_checkpoint_saved: path=%s acc=%.3f", output_path, acc)
    return {"loss": round(final_loss, 4), "train_acc": round(acc, 4)}


__all__ = ["train_re_checkpoint"]
