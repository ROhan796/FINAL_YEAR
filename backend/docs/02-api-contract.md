# API Contract

Every endpoint the frontend needs. Response shapes match `lib/mock-data.ts` interfaces **field-for-field** — no component changes required when wiring.

- Base: `/api/v1` (new, versioned)
- Legacy: `/api/*` — the 5 existing AI routes, **paths and JSON unchanged**
- Auth: `Authorization: Bearer <jwt>`; in dev, auto-login returns the seeded `Dr. Amara Nwosu`
- All list endpoints are cursor-paginated: `?cursor=&limit=` → `{items, nextCursor}`
- Errors: `{detail: string, code?: string, requestId: string}` with proper HTTP status

---

## 1. Auth / Users

| Method | Path | Notes |
|---|---|---|
| POST | `/auth/login` | `{email, password}` → `{accessToken, refreshToken, user}` |
| POST | `/auth/register` | → same shape |
| POST | `/auth/dev-login` | dev only — returns seeded user |
| POST | `/auth/refresh` | → new access token |
| GET | `/users/me` | → `UserProfile` |
| PATCH | `/users/me` | `{name?, title?, institution?, bio?, interests?, avatar?}` → `UserProfile` |
| PATCH | `/users/{id}/verified` | admin — `{verified: bool}` |
| POST | `/verification-requests` | `{institutionalEmail, orcid}` → `{id, status}` |
| GET | `/verification-requests/{id}` | → `{id, status, decidedAt?}` |

```ts
UserProfile { id, name, title, institution, field, bio, avatar, verified, interests[], email }
```

---

## 2. Papers / Catalog

| Method | Path | Query / Body |
|---|---|---|
| GET | `/papers` | `?q=&tag=&year=&minCitations=&openAccess=&sort=relevance\|citations\|newest&cursor=&limit=` |
| GET | `/papers/{id}` | → `Paper` |
| GET | `/papers/{id}/related` | vector neighbours — `Paper[]` |
| GET | `/papers/{id}/citation` | `?format=APA\|MLA\|BibTeX\|RIS` → `{citation}` |
| POST | `/papers/{id}/save` | idempotent |
| DELETE | `/papers/{id}/save` | idempotent |
| GET | `/saved` | `?query=&cursor=` → `Paper[]` |
| POST | `/papers/{id}/ai-analysis` | recompute + cache → `{aiSummary, keyContributions, methodology, limitations, researchGaps, citationContext}` |

```ts
Paper {
  id, title, authors[], abstract, year, venue, doi, tags[], citationCount,
  openAccess, pdfUrl?,
  relevanceSignal?: 'High'|'Medium'|'Low', reasoningEvidence?: string,
  aiSummary?, keyContributions?, methodology?, limitations?, researchGaps?, citationContext?
}
```

---

## 3. Search

| Method | Path | Body |
|---|---|---|
| POST | `/search` | `{query, mode: 'semantic'\|'keyword'\|'hybrid', filters?, cursor?, limit?, explain?}` |

**Does not accept the paper corpus** (the legacy `/api/search` still does, for compatibility — the server owns the index).

```jsonc
// 200
{
  "results": [Paper + relevanceSignal + reasoningEvidence],
  "query": "...", "mode": "hybrid",
  "queryUnderstanding": "1-sentence intent",
  "latencyMs": 41,
  "suggestedSearches": ["...", "..."],
  "total": 128, "nextCursor": "..."
}
```

| Method | Path | Notes |
|---|---|---|
| GET | `/search/history` | → `string[]` (max 10, deduped case-insensitive) |
| POST | `/search/history` | `{query}` |
| DELETE | `/search/history` | clear all |
| GET | `/search/suggestions` | personalized from history + interests → `string[]` |

Filters: `{year?, openAccess?, minCitations?, tag?}` — **server-side**, replacing the client-side filtering in `SearchView`.

---

## 4. Feed

| Method | Path | Body / Query |
|---|---|---|
| GET | `/feed` | `?sort=recency\|recommended&cursor=` → `FeedPost[]` |
| GET | `/posts/{id}` | permalink (enables real Share) |
| POST | `/posts` | `{content, postType, linkedPaperId?}` — requires `verified` |
| POST | `/posts/{id}/like` | idempotent |
| DELETE | `/posts/{id}/like` | idempotent |
| GET | `/posts/{id}/comments` | → `FeedComment[]` |
| POST | `/posts/{id}/comments` | `{content}` |
| DELETE | `/comments/{id}` | author only |

```ts
FeedPost {
  id, authorId, authorName, authorTitle, authorInstitution, authorAvatar, authorVerified,
  content, postType, timestamp, likes, linkedPaper?, comments[], tags[], isLiked?
}
FeedComment { id, authorName, authorInstitution, authorAvatar, content, timestamp, verified }
```

`isLiked` is **per-viewer**, computed server-side (the DB stores likes; the response sets the flag).

---

## 5. Network / Researchers

| Method | Path | Query |
|---|---|---|
| GET | `/researchers` | `?field=&q=&cursor=` → `Researcher[]` |
| GET | `/researchers/{id}` | → `Researcher` (enables the unwired profile view) |
| POST | `/researchers/{id}/follow` | idempotent |
| DELETE | `/researchers/{id}/follow` | idempotent |
| GET | `/researchers/{id}/followers` | → `Researcher[]` |
| GET | `/researchers/{id}/following` | → `Researcher[]` |

```ts
Researcher {
  id, name, title, institution, field, bio, avatar, verified,
  publicationsCount, citationCount, hIndex, followersCount,
  interests[], publications[], isFollowed?
}
```

`followersCount` computed from the `follows` table — no denormalised counter drift.

---

## 6. Collections

| Method | Path | Body |
|---|---|---|
| GET | `/collections` | → `Collection[]` |
| POST | `/collections` | `{name, description, paperId?}` — **`paperId` fixes the "Create & Add" bug** |
| GET | `/collections/{id}` | → `Collection` |
| PATCH | `/collections/{id}` | `{name?, description?, pinned?}` (rename/pin now possible) |
| DELETE | `/collections/{id}` | |
| POST | `/collections/{id}/papers` | `{paperId}` — 409 on duplicate |
| DELETE | `/collections/{id}/papers/{paperId}` | |
| POST | `/collections/{id}/ai-overview` | → `{summary}` (cross-paper synthesis; returns **both** `summary` and `executiveSummary` to fix the dead feature) |
| GET | `/collections/{id}/export` | `?format=bib\|ris\|md` |

```ts
Collection { id, name, description, pinned, paperIds[], createdDate, aiSummary?, shareableLink? }
```

---

## 7. Workspace / Projects

| Method | Path | Body |
|---|---|---|
| GET | `/projects` | → `WorkspaceProject[]` |
| POST | `/projects` | `{name, description, field}` |
| GET | `/projects/{id}` | → `WorkspaceProject` |
| DELETE | `/projects/{id}` | |
| POST | `/projects/{id}/members` | `{email, role}` — creates member **and** a notification |
| DELETE | `/projects/{id}/members/{email}` | |
| POST | `/projects/{id}/papers` | `{paperId}` |
| DELETE | `/projects/{id}/papers/{paperId}` | |
| GET/POST | `/projects/{id}/tasks` | `{title, dueDate, assigneeName}` |
| PATCH | `/projects/{id}/tasks/{taskId}` | `{status}` |
| DELETE | `/projects/{id}/tasks/{taskId}` | |
| GET/POST | `/projects/{id}/notes` | `{title, content, paperId?}` |
| PATCH/DELETE | `/projects/{id}/notes/{noteId}` | |
| GET/POST | `/projects/{id}/discussions` | `{content}` |
| DELETE | `/projects/{id}/discussions/{msgId}` | |
| GET/POST | `/projects/{id}/datasets` | `{name, description, format, size, link}` |
| DELETE | `/projects/{id}/datasets/{datasetId}` | |
| GET | `/projects/{id}/analytics` | real time-entries (replaces hardcoded 31 hrs + chart) |
| GET | `/projects/{id}/ai-thread` | persisted chat history (fixes lost-on-tab-switch) |
| DELETE | `/projects/{id}/ai-thread` | clear |

```ts
WorkspaceProject {
  id, name, description, field, status, ownerName, members[{name, role, email, avatar}],
  paperIds[], notes[], tasks[], milestones[], datasets[], documents[], discussions[], activityHistory[]
}
Task { id, title, assigneeName, assigneeAvatar, dueDate, status: 'Todo'|'In Progress'|'Done' }
```

Roles enforced: `Owner` > `Editor` > `Viewer` (viewer cannot mutate).

---

## 8. Publishing

| Method | Path | Body |
|---|---|---|
| GET | `/manuscripts` | → `ManuscriptSubmission[]` |
| POST | `/manuscripts` | full submission — **requires `verified`** |
| GET | `/manuscripts/{id}` | |
| POST | `/manuscripts/{id}/file` | `multipart/form-data` PDF — real upload to object storage |
| POST | `/manuscripts/{id}/ai-prereview` | → `{aiReview}` (replaces fake filename) |
| PATCH | `/manuscripts/{id}/status` | `{status}` — reviewer/admin |

```ts
ManuscriptSubmission {
  id, title, abstract, authors[], coAuthors[], venue, tags[], year,
  fileName, fileSize, integrityConfirmed,
  status: 'Draft'|'Under AI Review'|'Submitted'|'Published'|'Revision Required',
  submittedAt, aiReview?
}
```

---

## 9. Notifications

| Method | Path |
|---|---|
| GET | `/notifications` | `?unreadOnly=&cursor=` → `NotificationItem[]` |
| GET | `/notifications/unread-count` | → `{count}` (Navbar badge) |
| PATCH | `/notifications/{id}` | mark read |
| POST | `/notifications/read-all` | |

```ts
NotificationItem { id, type: 'mention'|'comment'|'collaboration'|'paper_update'|'verification',
                   title, message, timestamp, read, link? }
```

Server emits notifications on: new comment on your post, follow, project invite, mention, verification decision, manuscript status change.

---

## 10. Trending & Graph

| Method | Path | Notes |
|---|---|---|
| GET | `/trending` | → `TrendingTopic[]`; recomputed by a scheduled job |
| GET | `/graph/exploration` | `?nodeId=&hops=2&relation=` → nodes/edges (Neo4j) |
| GET | `/discoveries` | ranked cross-domain link predictions → `{score, srcNode, relation, dstNode, evidencePaths[], papers[]}` |
| POST | `/discoveries/{id}/feedback` | `{useful: bool}` — retrains ranking |

---

## 11. Legacy AI routes (UNCHANGED paths + shapes)

Mounted at `/api/*` so the current frontend works as-is.

| Path | In | Out |
|---|---|---|
| `POST /api/search` | `{query, mode, papers?, isHttpAccelerated?}` | `{results, query, mode, queryUnderstanding, latencyMs, suggestedSearches, isHttpAccelerated}` |
| `POST /api/summarize` | `{title, abstract, authors?, venue?}` | `{summary}` |
| `POST /api/generate-insights` | `{title, abstract, authors?, tags?}` | `{keyContributions, methodology, limitations, researchGaps, citationContext}` |
| `POST /api/analyze-draft` | `{title, abstract?, venue?, tags?, fileName?, draftText?}` | `{aiReview:{predictedScore, titleReview, executiveCritique, suggestedKeywords, likelyPeerQuestions, clarityAssessment, strengths, weaknesses}}` |
| `POST /api/chat-assistant` | `{message, projectContext?, paperContext?, conversationHistory?}` | `{reply}` |

Backed by: hybrid retrieval (search), Gemini with deterministic fallback (the rest). Every response validated against a Pydantic model so a provider change can never break the frontend.

---

## 12. Health / Ops

| Path | Purpose |
|---|---|
| `GET /health/live` | process up |
| `GET /health/ready` | Postgres + Redis reachable, migrations current |
| `GET /api/v1/meta` | version, feature flags (ml_enabled, neo4j_enabled) |
