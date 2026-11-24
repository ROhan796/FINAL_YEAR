# CKGAI ML Pipeline & Novel Algorithms

> Technical reference for the 7-layer Cross-Domain Knowledge Graph AI (`backend/ml/`) and the 7 novel research-intelligence algorithms.

## 1. Seven-Layer ML Architecture

| Layer | Module | Model / Algorithm | Checkpoint |
|---|---|---|---|
| 1. Ingestion | `ml/ingest/` | arXiv Atom API, PubMed E-utilities, Semantic Scholar Graph API | Stateless HTTP adapters |
| 2. Preprocessing | `ml/preprocess/` | LaTeX/HTML cleanup, sentence splitting, chunking (1200 chars, 150 overlap) | — |
| 3. Encoders & MRL | `ml/encoders/` | SPECTER2 (`allenai/specter2_base`, 768-d), SciBERT (`allenai/scibert_scivocab_uncased`), Matryoshka 128-d/768-d | HuggingFace cache |
| 4A. NER | `ml/ner/` | SciBERT + token classification head + Linear-Chain CRF (17 BIO tags, 8 types) + Gazetteer supplement | `models/checkpoints/ner_head.pt` |
| 4B. Relation & Polarity | `ml/relation/` | SciBERT pair MLP `[h_e1 ‖ h_e2 ‖ h_e1⊙h_e2 ‖ h_cls]` + type constraints; 3-class Citation Polarity | `re_head.pt`, `citation_polarity.pt` |
| 5. Knowledge Graph | `ml/graph/` | `EntityResolver` (exact + cosine > 0.92), `KGBuilder` (provenance), `Neo4jProjector` (`updated_at` watermark) | Postgres (`kg_nodes`, `kg_edges`) |
| 6. PyTorch GNN | `ml/gnn/` | `CKGAI_GNN` (`L=3`, `h=256`, R-GCN `B=30`, GAT `K=8`, `VirtualNode`), AdamW + CosineAnnealing + Margin/BCE | `models/checkpoints/ckgai_gnn.pt` |
| 7. Discovery & Calibration | `ml/discovery/` | DistMult / TransE / Dot / MLP scorers, cross-domain boost `λ=1.3`, Platt logistic calibration | `discovery_calibrator.json` |

## 2. Novel Algorithms Implemented

1. **Citation Polarity Classification (6.A)** — `ml/relation/citation_polarity.py`
   3-class (`supports`, `disputes`, `neutral`) classifier combining SciBERT pair embeddings + MLP head with explicit stance cue signals. Exposed at `POST /api/v1/papers/{id}/citation-polarity`.
2. **Research Gap Detection (6.B)** — `app/modules/insights/service.py:research_gaps`
   Cross-domain co-occurrence analysis over `kg_nodes` and `kg_edges` identifying `(METHOD | ALGORITHM) × (DISEASE | MATERIAL | PHENOMENON | DATASET)` pairs unexplored across domains. Exposed at `GET /api/v1/insights/research-gaps`.
3. **Temporal Trend Embedding Boost (6.C)** — `app/modules/search/service.py:_signals`
   Applies `final *= (1 + 0.12 * sigmoid(year - 2020))` on top of RRF + cosine + citation prior.
4. **Reading-Order Graph & Gateway Papers (6.D)** — `app/modules/insights/service.py:gateway_papers`
   Constructs a directed transition graph from `saved_papers` timestamps and `related_papers` links, running 40-iteration PageRank (`d=0.85`) to surface foundational gateway papers.
5. **Contrastive Collection Clustering (6.E)** — `app/modules/insights/service.py:collection_clusters`
   Runs `scikit-learn` K-Means (`k = round(sqrt(n))`) over stored paper embeddings and labels each cluster using top TF-IDF n-grams without an LLM call.
6. **Confidence-Calibrated Discovery Scores (6.F)** — `ml/discovery/calibration.py:PlattCalibrator`
   Fits logistic calibration parameters from `discoveries.useful_votes` and `dismissed` feedback; automatically recalibrates on every feedback vote and via `POST /api/v1/discoveries/retrain`.
7. **Matryoshka Embeddings (MRL) (6.G)** — `ml/encoders/matryoshka.py`
   Provides `truncate_and_normalize` (128-d/256-d/512-d/768-d), `matryoshka_contrastive_loss`, and `two_stage_matryoshka_rank` (128-d coarse shortlist → 768-d exact cosine rerank) wired into `SearchRepository._vector_rank`.

## 3. CLI Operations

```bash
cd backend
.venv\Scripts\python.exe -m app.cli seed
.venv\Scripts\python.exe -m app.cli ingest --source arxiv --query "graph neural networks" --max 50 --embed --kg
.venv\Scripts\python.exe -m app.cli index-kg --limit 200
.venv\Scripts\python.exe -m app.cli train-gnn --epochs 30
.venv\Scripts\python.exe -m app.cli discover --train
.venv\Scripts\python.exe -m app.cli project-neo4j
```
