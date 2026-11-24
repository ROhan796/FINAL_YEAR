# AI Scientific Paper Knowledge Engine — System Architecture

**CKGAI · Cross-Domain Knowledge Graph AI**
Backend + ML system design for the AI-Native Research Intelligence Platform.

| | |
|---|---|
| Frontend | Next.js 15 / React 19 / Tailwind 4 (`../`) — **complete, not modified in this pass** |
| Backend | FastAPI (Python 3.11), modular monolith |
| OLTP | PostgreSQL 16 (NeonDB) — source of truth |
| Vectors | pgvector (in the same Postgres) |
| Graph | Neo4j 5 — **projected read model**, Postgres is source of truth |
| Cache / Queue | Redis 7 |
| LLM | Gemini (provider-abstracted) |
| ML | PyTorch + PyTorch Geometric, SciBERT / SPECTER2 |

---

## 1. What Exists Today (Frontend Index)

The frontend is a **social network for researchers** with an **ML-powered search bar**. It is fully built, runs on `localStorage`, and speaks to five prototype AI routes. Nothing else has a backend.

### 1.1 Route map

| Route | View | Domain |
|---|---|---|
| `/` | `FeedView` | Social feed |
| `/search` | `SearchView` | Hybrid search + AI query understanding |
| `/discover` | `DiscoverView` | Corpus browsing, tag/sort filters |
| `/trending` | `TrendingView` | Emerging topic ranking |
| `/network` | `NetworkView` | Researcher directory + follow |
| `/collections` | `CollectionsView` | Collection CRUD + AI synthesis |
| `/saved` | `SavedPapersView` | Saved library |
| `/workspace` | `WorkspaceView` | Projects, tasks, notes, datasets, discussion, AI chat |
| `/publish` | `PublishView` | Manuscript pipeline + AI pre-review |
| `/notifications` | `NotificationsView` | Activity inbox |
| `/settings` | `SettingsView` | Profile + verification |

Seven modals are mounted globally in `ClientShell`: `PaperDetailModal`, `CiteModal`, `AddToCollectionModal`, `NewPostModal`, `NewProjectModal`, `NewManuscriptModal`, `VerificationModal`.

### 1.2 Entity model (frontend contract)

`Paper`, `Researcher`, `FeedPost`, `FeedComment`, `Collection`, `WorkspaceProject` (+ `Task`, `ProjectNote`, `ProjectDataset`, `ProjectDocument`, `ProjectDiscussion`, `Milestone`), `NotificationItem`, `TrendingTopic`, `ManuscriptSubmission`, `UserProfile`.

These interfaces in `lib/mock-data.ts` are the **authoritative wire format**. The backend returns them field-for-field so the frontend can be swapped from `localStorage` to `fetch` without touching a single component.

### 1.3 Existing AI contracts that must not break

| Endpoint | Request | Response |
|---|---|---|
| `POST /api/search` | `{query, mode, papers, isHttpAccelerated}` | `{results, query, mode, queryUnderstanding, latencyMs, suggestedSearches}` |
| `POST /api/summarize` | `{title, abstract, authors?, venue?}` | `{summary}` |
| `POST /api/generate-insights` | `{title, abstract, authors?, tags?}` | `{keyContributions, methodology, limitations, researchGaps, citationContext}` |
| `POST /api/analyze-draft` | `{title, abstract?, venue?, tags?, fileName?, draftText?}` | `{aiReview:{predictedScore, titleReview, executiveCritique, suggestedKeywords, likelyPeerQuestions, clarityAssessment, strengths, weaknesses}}` |
| `POST /api/chat-assistant` | `{message, projectContext?, paperContext?, conversationHistory?}` | `{reply}` |

All five are re-implemented at the **same paths with the same JSON shapes**, but backed by the real ML stack instead of a prompt-only heuristic.

### 1.4 User types

There is **no auth today** (`README` mentions Clerk; no `middleware.ts`, no import). The design adds real roles so the portal can differentiate users, while shipping a **dev auto-login** so the current UI works immediately.

| Role | Can |
|---|---|
| `reader` | Browse, search, save, follow, comment |
| `researcher` | Everything above + post, create collections/projects, submit manuscripts |
| `verified_researcher` | Above + publish to feed, submit manuscripts, verified badge |
| `editor` / `reviewer` | Manuscript review queue |
| `admin` | Verification approvals, moderation, ingestion control |

**Single-portal, multi-role**: one UI, capability gates (`user.verified` is already the gate for `NewPostModal` and `NewManuscriptModal`).

---

## 2. Top-Level Architecture

Two planes. Confusing them is the main way this kind of system becomes unscalable.

```
                            ┌──────────────────────────────────────────┐
   Browser (Next.js) ─────► │           API GATEWAY / Ingress          │
                            │   TLS · rate limit · auth · request ID   │
                            └────────────────┬─────────────────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
              ▼                              ▼                              ▼
   ┌─────────────────────┐       ┌─────────────────────┐        ┌─────────────────────┐
   │  FASTAPI  (stateless│       │  FASTAPI  (stateless│  ...   │  FASTAPI  (stateless│
   │  xN, horizontal)    │       │  xN, horizontal)    │        │  xN, horizontal)    │
   └──────────┬──────────┘       └──────────┬──────────┘        └──────────┬──────────┘
              │     ONLINE PLANE (p95 < 300ms)                            │
              │                                                            │
   ┌──────────▼──────────────────────────────────────────────────────────────┐
   │                      Redis (cache · sessions · rate limit · pubsub)     │
   └──────────┬───────────────────────────────┬──────────────────────────────┘
              │                               │
   ┌──────────▼──────────────┐     ┌──────────▼─────────────┐
   │  PostgreSQL (NeonDB)    │     │  Neo4j (Aura / server) │
   │  · OLTP  · pgvector     │     │  projected knowledge   │
   │  · FTS (tsvector)       │     │  graph, READ-ONLY      │
   │  SOURCE OF TRUTH        │     │  from projector        │
   └──────────▲──────────────┘     └──────────▲─────────────┘
              │        OFFLINE PLANE          │
   ┌──────────┴───────────────────────────────┴──────────────────────────────┐
   │          Task queue (Redis / Arq)  ·  workers (CPU)  ·  GPU jobs        │
   ├─────────────────────────────────────────────────────────────────────────┤
   │  Ingest → Preprocess → Encode → NER → RE → KG build → Project → GNN    │
   └─────────────────────────────────────────────────────────────────────────┘
              ▲
              │  arXiv · PubMed · Semantic Scholar
```

### 2.1 Why this split

| Plane | Workload | Latency budget | Scaling lever |
|---|---|---|---|
| **Online** | CRUD, feed, hybrid search, RAG chat, summaries | p95 < 300 ms | Horizontal replicas of stateless FastAPI |
| **Offline** | Ingestion, embedding, NER/RE, KG build, GNN training, discovery ranking | minutes → hours | Queue depth + worker count + GPU |

**Rule:** no request handler ever runs a Transformer forward pass over a batch, and no handler trains anything. Heavy work is enqueued; the handler returns fast. The single exception is *single-query* embedding (one text → one vector, ~5 ms on GPU, ~30 ms on CPU) which is cached.

### 2.2 Why a modular monolith (not microservices)

- One transaction boundary — a `POST /collections` that creates a collection and links a paper is atomic.
- One connection pool, one deploy, one set of migrations.
- Every module lives in `app/modules/<name>/` with `router.py` + `service.py` + `repository.py`, so any module can be **extracted into its own service later** by changing only its imports — the seams already exist.
- At project scale, microservices add distributed-debugging cost with no benefit.

---

## 3. Data Layer

### 3.1 Storage decisions

| Store | Owns | Why |
|---|---|---|
| **Postgres (NeonDB)** | All OLTP entities, embeddings (`vector`), full-text (`tsvector`), knowledge-graph nodes/edges | One ACID boundary; Neon scales to zero for dev, branches for preview; pgvector keeps vectors **in the same transaction** as the row they describe (no dual-write) |
| **pgvector** | Paper/chunk/user/query embeddings, HNSW index | User requirement; correct choice at ≤10M vectors. Beyond that, spill to a dedicated ANN store behind the same repository interface |
| **Neo4j** | Projected graph for multi-hop exploration + GNN sampling | Matches the CKGAI spec; used as a **read model** so there is never a dual-write consistency problem |
| **Redis** | Query cache, embedding cache, rate limits, job queue, feed fan-out | Sub-ms reads; prevents cache stampede with per-key locking |
| **Object storage (S3/R2)** | Manuscript PDFs, datasets | Never put binaries in Postgres |

### 3.2 The dual-database rule (Postgres + Neo4j)

This is the one place the design has two databases for one concept, so the rule is explicit:

1. **Postgres is the only writer.** `kg_nodes` / `kg_edges` tables are the source of truth, written inside the same transaction as paper ingestion.
2. **Neo4j is written by exactly one thing:** the *graph projector*, which tails a `kg_edges.updated_at` watermark and upserts into Neo4j. Idempotent, resumable, replayable from zero.
3. **Online reads go to Postgres** (1–3 hops, indexed, same pool, same transaction).
4. **Neo4j serves two consumers only:**
   - the **exploration UI** (`GET /api/v1/graph/exploration`) where unbounded Cypher patterns are needed;
   - the **GNN sampler**, which bulk-exports neighbourhoods for training.
5. If Neo4j is down, **the product still works** — only deep exploration degrades. Reconcile by re-running the projector.

```
  ingestion tx ──► kg_nodes/kg_edges (Postgres, truth)
                        │
                        │  watermark = max(updated_at)
                        ▼
                 graph projector (idempotent upsert)
                        │
                        ▼
                   Neo4j (read model)
                     ├────► exploration API
                     └────► GNN neighbourhood sampler
```

### 3.3 Schema outline

**Application**

```
users, user_profiles, verification_requests
papers, paper_authors, paper_tags, paper_embeddings
saved_papers, search_history
collections, collection_papers
feed_posts, feed_post_tags, feed_comments, post_likes
follows (follower_id, followee_id)
researcher_profiles
projects, project_members, project_notes, project_tasks,
project_datasets, project_documents, project_discussions, project_papers
manuscripts, manuscript_ai_reviews
notifications
trending_topics, trending_topic_papers
```

**Knowledge graph (Postgres, source of truth)**

```sql
kg_nodes  (id uuid pk, canonical_name, entity_type, domain, embedding vector(768),
           paper_id fk, confidence numeric, created_at, updated_at)
kg_edges  (id uuid pk, src_id fk, dst_id fk, relation text, confidence numeric,
           provenance_paper_id fk, provenance_sentence text, domain_cross boolean,
           created_at, updated_at)
```

Indexes: `kg_edges (src_id, relation)`, `(dst_id, relation)`, `(updated_at)` (projector watermark), `kg_nodes (entity_type, domain)`, HNSW on both embedding columns.

**Embedding dimensions**

| Column | Model | Dim |
|---|---|---|
| `papers.embedding` | SPECTER2 | 768 |
| `paper_chunks.embedding` | SPECTER2 / SciBERT CLS | 768 |
| `kg_nodes.embedding` | SciBERT | 768 |
| `queries.embedding` (ephemeral, cached) | SPECTER2 | 768 |

Single dimension (768) across the board keeps `pgvector` indexes uniform and lets one ANN index serve paper search, chunk RAG and entity resolution.

---

## 4. Search — the core ML feature

Three modes the UI exposes: `semantic`, `keyword`, `hybrid`. They are **real**, not echoes.

### 4.1 Retrieval path

```
query q
  │
  ├─► Redis query-embedding cache ──(miss)──► encoder.encode(q) ──► 768-d
  │
  ├─► A. Vector arm:   papers.embedding  @ cosine, HNSW, top-K=100
  ├─► B. FTS arm:      papers.search_tsv @@ plainto_tsquery, top-K=100
  ├─► C. KG arm (hybrid only): entity link q → kg_nodes → 1-hop paper neighbours
  │
  └─► Reciprocal Rank Fusion ──► filters (year, openAccess, minCitations, tag)
              │
              ▼
        top-N + per-result {relevanceSignal, reasoningEvidence}
```

**Mode mapping**

| Mode | Arms used |
|---|---|
| `semantic` | A only |
| `keyword` | B only |
| `hybrid` | A + B (+ C when entities resolve), **RRF** |

**Reciprocal Rank Fusion** (robust — no score normalisation needed between cosine and BM25):

```
RRF(d) = Σ_arm  1 / (k + rank_arm(d))        k = 60
```

### 4.2 Why SPECTER2 for paper embeddings (improvement on the spec)

The spec uses **SciBERT** for everything. SciBERT is a *language* encoder — its `[CLS]` vector is a poor sentence similarity representation out of the box (BERT's anisotropy). For *paper-level* retrieval the correct model is:

| Model | Used for | Why |
|---|---|---|
| **`allenai/specter2_base`** | Paper-level vectors (search, related-papers, entity resolution) | Trained on **citation links** — two papers that cite each other are close even with disjoint vocabularies. Exactly the "vocabulary mismatch" problem §2.3 of the spec calls out |
| **`allenai/scibert_scivocab_uncased`** | NER, RE, passage/chunk encoding, `kg_nodes` vectors | Per the spec; the right model for token-level tasks |
| **`microsoft/BiomedNLP-PubMedBERT-base`** | Optional biomedical reranker | Domain swap behind the same interface |

SPECTER2 is an **addition**, not a replacement — SciBERT keeps its specified role. The encoder layer is a registry (`ml/encoders/`), so this is one config line.

### 4.3 Relevance signals

Each result carries `relevanceSignal ∈ {High, Medium, Low}` and a `reasoningEvidence` sentence — the exact shape the frontend renders. Computed as:

```
final = 0.65 · rrf_normalized + 0.20 · cosine + 0.15 · citation_prior(year, citations)
signal: High ≥ 0.70 · Medium ≥ 0.40 · else Low
evidence: generated from matched entities + matched terms + retrieval arm
```

Evidence is **deterministic and traceable** (which arm, which terms, which KG entities) — better than asking an LLM to invent a reason per paper, and it costs zero latency. The LLM path stays as an optional `explain=true` enrichment.

### 4.4 Latency budget (hybrid, p95)

| Step | Budget |
|---|---|
| Query encode (cached) | 0 ms / 30 ms miss |
| HNSW + FTS + filters (parallel) | 40 ms |
| RRF + hydrate top-N | 20 ms |
| Evidence build | 5 ms |
| **Total** | **< 100 ms** |

`isHttpAccelerated` from the UI is preserved for contract compatibility (it divides reported latency), but the real number is always reported.

---

## 5. Chat / Assistant — how it scales

The workspace assistant is the most chat-like feature, so it gets an explicit design.

### 5.1 RAG path

```
POST /api/chat-assistant {message, projectContext, conversationHistory}
   │
   ├─ 1. persist turn  (chat_messages table — history stops being client-local)
   ├─ 2. build retrieval set:
   │       • project.paperIds  → paper chunks (pinned, always included)
   │       • message embedding → hybrid search over project's papers (top-8)
   ├─ 3. Redis prompt cache: key = hash(system + pinned_ids + message)
   ├─ 4. LLM call (Gemini) — streaming (SSE)
   └─ 5. cache answer, return {reply}
```

### 5.2 Scaling levers

1. **Stateless handlers** — `conversationHistory` is accepted for backward compatibility, but the server owns history in `chat_messages`, so a load balancer can route any turn to any replica.
2. **Streaming (SSE)** — first token in < 700 ms regardless of answer length; connection cost moves from *duration* to *time-to-first-token*.
3. **Prompt caching** — Redis key on `(system_prompt_version, project_id, pinned_paper_set)`. Repeat questions about the same project skip retrieval.
4. **Bounded context** — history trimmed to last N turns + summary; retrieval hard-capped at 8 chunks × ~500 tokens. Context size is the #1 cost/latency driver and it is explicitly bounded.
5. **Rate limiting** — per-user token bucket in Redis; LLM calls are the most expensive thing in the system.
6. **Provider abstraction** — `LLMProvider` interface with Gemini/OpenAI/llama-cpp implementations; a failed provider degrades to a deterministic fallback (as the frontend does today) instead of a 500.

### 5.3 Fan-out for social feed

Read-heavy, write-light — classic **pull with cache** (no fan-out-on-write at this scale):

```
GET /feed?sort=recency|recommended
  Redis key feed:{user_id}:{sort}:{page}
    miss → SQL: posts of followed authors ∪ own, ordered
            → recommended = ml ranking (engagement + embedding similarity to user interests)
            → hydrate authors, papers, like state, comment counts
            → SETEX 60s
```

Invalidate on `POST /posts` / like / comment by deleting the affected user's keys (a `DEL` on a pattern via a Redis keyspace notification, or a version-counter key). Cache-aside beats fan-out tables until ~10⁵ followers per user.

---

## 6. CKGAI ML Pipeline (modular)

Implements the spec's seven layers as **independently runnable, independently testable** packages under `ml/`.

```
ml/
├── config.py              all hyperparameters (from the spec's appendix table)
├── ingest/                Layer 1   arxiv · pubmed · semantic_scholar
├── preprocess/            Layer 2   normalize · segment · tokenize · sections
├── encoders/              Layer 3   EncoderBase · SciBERT · SPECTER2 · registry
├── ner/                   Layer 4A  NERBase · TransformerNER · CRF · gazetteer
├── relation/              Layer 4B  RelationExtractor · pair builder · classifier
├── graph/                 Layer 5   schema · entity_resolution · builder · repos
├── gnn/                   Layer 6   GCN · GAT · RGCN · dataset · train
├── discovery/             Layer 7   scorers (dot/distmult/transE/mlp) · ranking
├── pipelines/             orchestration  ingest · index · discovery
└── serving/               retrieval · rag · embed_store  (online surface)
```

Every layer:
- inherits a **protocol/base class** (swap SciBERT for PubMedBERT by config),
- takes plain dataclasses in / out (unit-testable with no GPU),
- lazy-imports `torch` so the **API boots without the ML extra installed**.

### 6.1 Layer 3 — Encoder

```python
class EncoderBase(Protocol):
    dim: int
    def encode(self, texts: list[str], *, kind: str) -> np.ndarray: ...
```
`kind ∈ {paper, passage, entity, query}` routes to the right pooling (SPECTER2 title+abstract for `paper`, SciBERT mean-pool for `passage`, CLS for `entity`). Batched, GPU, LRU-cached.

### 6.2 Layer 4 — NER + RE

- **NER:** SciBERT token-classification head, BIO tags, 8 entity types from the spec (`METHOD, MATERIAL, DISEASE, ALGORITHM, DATASET, PHENOMENON, METRIC, CHEMICAL`) → 17 tags + `O`. Linear-chain **CRF** head enforces legal BIO transitions (Viterbi at decode).
- **RE:** entity-pair vector `[h_e1 ‖ h_e2 ‖ h_e1⊙h_e2 ‖ h_cls]` → MLP over 8 relations + `no_relation`, with **type constraints** (a `DATASET` can't `treat` a `METHOD`) applied before softmax — cuts false positives and shrinks the candidate set.
- Datasets: **SciERC** (NER+RE), **BC5CDR**, **JNLPBA**.

### 6.3 Layer 5 — Graph construction

- `EntityResolver` merges surface forms: normalise → exact match → cosine > 0.92 → ontology lookup (MeSH/UMLS when configured).
- Every edge stores `provenance_paper_id` + `provenance_sentence` + `confidence` → **every discovery is traceable to evidence**, which the spec flags as a key contribution.
- Writes to Postgres `kg_nodes`/`kg_edges` in one transaction, then the projector syncs Neo4j.

### 6.4 Layer 6 — GNN

Spec is followed exactly: **L = 3**, hidden 256, **R-GCN primary** with **basis decomposition B = 30**, **GAT K = 8** heads, dropout 0.1.

Improvement: **relation-aware attention + virtual node.**
- R-GCN alone under-weights rare relations; GAT on top lets each head specialise (one head learns to attend to same-domain neighbours, another to cross-domain).
- A **virtual global node** connected to all nodes gives every embedding a global context signal in 1 hop, so 3 hops reach effectively the whole graph without going deeper (avoids the over-smoothing the spec correctly warns about).

Training: BCE + margin ranking (`γ = 1.0`), type-constrained negative sampling (5 per positive), Adam `η=1e-3` + cosine annealing, grad clip 1.0, batch 1024. Eval: **Hits@10, MRR, AUC-ROC, P@K**. Targets per spec: Hits@10 > 0.85, MRR > 0.4.

### 6.5 Layer 7 — Discovery ranking

Four scorers behind one interface, selected by config — **DistMult is the default** (bilinear, relation-aware, cheap at inference):

| Scorer | Score | Use |
|---|---|---|
| Dot | `zᵢᵀzⱼ` | baseline |
| **DistMult** | `zᵢᵀ diag(r) zⱼ` | **default** — relation-specific |
| TransE | `-‖zᵢ + r − zⱼ‖₂` | compositional relations (`derived_from`) |
| MLP | `MLP([zᵢ ‖ r ‖ zⱼ])` | highest expressiveness, slower |

Discovery boost (per spec):

```
discovery_score(vᵢ, r, vⱼ) = score(vᵢ,r,vⱼ) · 1[domain(vᵢ) ≠ domain(vⱼ)] · λ     λ = 1.3
```

Candidate generation is the scalability trap here — "all unconnected pairs" is O(N²). It is bounded three ways:
1. **Block by domain** — only cross-domain pairs (that is the product's whole point).
2. **ANN pre-filter** — HNSW nearest neighbours of each node, top-200, instead of the full cross product.
3. **Type rules** — only relation-legal type pairs are scored.

That turns O(N²) into roughly O(N · 200 · 0.1) and makes the engine runnable on a laptop corpus.

### 6.6 Which model for which job (summary)

| Job | Model | Store |
|---|---|---|
| Paper semantic search | **SPECTER2** | pgvector HNSW |
| Related papers | SPECTER2 cosine | pgvector |
| Chunk retrieval for RAG | SciBERT mean-pool | pgvector |
| Entity vectors / resolution | SciBERT CLS | `kg_nodes.embedding` |
| NER | SciBERT + CRF head | torch checkpoint |
| Relation extraction | SciBERT + pair MLP | torch checkpoint |
| Graph reasoning | R-GCN + GAT + virtual node | torch checkpoint |
| Discovery ranking | DistMult | torch checkpoint |
| Summaries / insights / pre-review / chat | **Gemini** (provider-abstracted) | Redis cache |

---

## 7. Scalability Design

### 7.1 API tier

- **Stateless** → horizontal replicas behind a load balancer. No server-side session.
- **Connection pooling**: Neon's built-in pooler for transaction mode; app-level `NullPool`/small pool for long queries. Never let N replicas × M conns exceed Postgres `max_connections`.
- **Read replicas** for `GET /papers`, `/feed`, `/discover` (eventual consistency is fine); all writes to primary.
- **Pagination everywhere** (keyset, not OFFSET) — `cursor` params.
- **Idempotency keys** on `POST /manuscripts`, `POST /posts`.

### 7.2 Database

| Technique | Applied to |
|---|---|
| HNSW index | `papers.embedding`, `paper_chunks.embedding`, `kg_nodes.embedding` |
| GIN + `tsvector` | full-text arm |
| Partial indexes | `WHERE deleted_at IS NULL`, `feed_posts (author_id, created_at DESC)` |
| Keyset pagination | feed, papers, notifications, search |
| Table partitioning (by month) | `feed_posts`, `search_logs`, `chat_messages` |
| `updated_at` watermark | graph projector (incremental, not full rebuild) |
| Connection pooling | PgBouncer / Neon pooler |

### 7.3 Offline / ML tier

- **Queue** (Redis + Arq/Celery): `ingest`, `embed`, `ner`, `re`, `project`, `train`, `discover` — separate queues so a GNN training job can never starve ingestion.
- **Backpressure**: worker concurrency per queue; ingest rate-limited to respect source APIs (arXiv 1 req/3 s).
- **Checkpointed stages** — each stage writes its output and a watermark, so a crash resumes rather than restarts.
- **Batch embedding** in chunks of 256 on GPU; incremental (only re-embed new/changed papers).
- **GNN training** offline; serving only loads the checkpoint for inference. Incremental graph updates re-run link prediction for affected subgraph, not full retrain.

### 7.4 Caching

| Key | TTL | Invalidate |
|---|---|---|
| `qemb:{hash}` query embedding | 24 h | never |
| `search:{hash}` results | 5 min | on corpus reindex |
| `feed:{user}:{sort}:{page}` | 60 s | post/like/comment |
| `paper:{id}` | 10 min | paper update |
| `ai:{endpoint}:{hash}` | 24 h | never (content-addressed) |
| `prompt:{project}:{hash}` | 10 min | project paper change |

Stampede protection: Redis `SET NX EX` lock per key; losers poll.

### 7.5 Rate limiting & abuse

Per-user and per-IP token buckets in Redis. AI endpoints get a tighter bucket (cost). Search gets a looser one.

---

## 8. Reliability & Observability

- **Structured JSON logs** with `request_id` propagated to workers.
- **OpenTelemetry** traces across API → DB → queue → worker.
- **Health**: `/health/live` (process), `/health/ready` (Postgres + Redis reachable, migrations current).
- **Graceful degradation matrix** — the system must degrade, not 500:

| Dependency down | Behaviour |
|---|---|
| Neo4j | Deep exploration unavailable; everything else fine |
| Redis | Cache bypass, rate limit falls back to in-process, queue pauses |
| Gemini | Deterministic fallback responses (matches current frontend behaviour) |
| GPU / ML checkpoints | Lexical + FTS search only; signals default to `Medium` |

- **Migrations** via Alembic, run as a release step, never at import.
- **Tests**: unit (services, scorers, RRF), contract (every response matches the frontend interface), integration (Testcontainers Postgres), ML (tiny synthetic graph — no GPU in CI).

---

## 9. Deployment

```
frontend  ── Vercel / static
backend   ── Docker image → Fly.io / Cloud Run / K8s (HPA on CPU + p95)
postgres  ── NeonDB (branch per env)
neo4j     ── Neo4j Aura (free tier) or docker
redis     ── Upstash / Elasticache
workers   ── same image, different command (arq worker), GPU node for train
storage   ── S3/R2
```

Environments: `dev` (docker-compose, Neo4j optional) → `staging` (Neon branch) → `prod`.

---

## 10. Module Map (implementation)

```
backend/
├── app/
│   ├── main.py                 FastAPI factory, CORS, lifespan, router mounting
│   ├── core/                   config · security · errors · logging · deps · pagination
│   ├── db/                     engine · session · pgvector types
│   ├── models/                 SQLAlchemy ORM (Postgres source of truth)
│   ├── schemas/                Pydantic wire contracts
│   ├── modules/
│   │   ├── auth/               JWT, roles, dev auto-login
│   │   ├── users/              profile, verification requests
│   │   ├── papers/             catalog, save, related, citation export, AI analysis
│   │   ├── search/             hybrid retrieval, history, suggestions
│   │   ├── feed/               posts, comments, likes, ranking
│   │   ├── network/            researchers, follow graph
│   │   ├── collections/        CRUD + membership
│   │   ├── workspace/          projects, tasks, notes, datasets, discussions, members, AI thread
│   │   ├── publishing/         manuscripts, file upload, AI pre-review, status
│   │   ├── notifications/      inbox, read state
│   │   ├── trending/           topic ranking
│   │   └── graph/              exploration API + projector trigger
│   ├── ai/                     providers/ (gemini, fallback) + 5 legacy contracts
│   └── api/                    v1 router + legacy /api router
├── ml/                         the seven CKGAI layers (see §6)
├── alembic/                    migrations (pgvector + all tables)
├── tasks/                      queue definitions + workers
└── tests/
```

Each module: `router.py` (HTTP only) → `service.py` (business rules, transactions) → `repository.py` (SQL only). `router` never touches SQL; `repository` never knows about HTTP.

---

## 11. Delivery Phases

| Phase | Scope |
|---|---|
| **P0** | Architecture docs, API contract, modular backend scaffold, pgvector schema, 5 AI contracts, ML package |
| **P1** | Frontend wiring (`platform-store` → API + connection state), JWT auth UI (`/login`, `/register`, `middleware.ts`), Landing `/` + `/feed`, Arq worker + CLI + OpenTelemetry |
| **P2** | Real arXiv/PubMed/S2 ingestion, SPECTER2+Matryoshka backfill, trained NER/RE/Citation-Polarity checkpoints, PyTorch GNN training, Platt-calibrated discoveries, Graph UI |
| **P3** | Neo4j incremental projector, RAG SSE streaming chat, Reading Room annotations, 7 novel algorithms + 7 out-of-the-box features, pytest + CI |
