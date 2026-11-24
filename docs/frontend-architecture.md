# Frontend Architecture — suiiiiii / CKGAI

> Next.js 15 (App Router) · React 19 · TypeScript · Tailwind CSS 4 · Recharts · Lucide React

## 1. Route Map & Public/Private Separation

| Route | Component / View | Access | Purpose |
|---|---|---|---|
| `/` | `LandingView` | Public | Product overview, workflow diagram, CTAs to `/feed`, `/graph`, `/search`, `/login` |
| `/login` | `LoginPage` | Public | Email/password JWT sign-in + quick Demo Researcher sign-in |
| `/register` | `RegisterPage` | Public | Researcher registration + profile creation |
| `/feed` | `FeedView` | Protected (`middleware.ts`) | Social research feed, likes, comments, follow suggestions |
| `/search` | `SearchView` | Protected | Hybrid semantic search, server-side filters, "Why This Paper?" provenance panel |
| `/discover` | `DiscoverView` | Protected | Corpus browser + Gateway Papers (Reading-Order PageRank) strip |
| `/graph` | `GraphView` | Protected | Interactive SVG Knowledge Graph, GNN Cross-Domain Discoveries, Research Gaps, Live Ingestion |
| `/workspace` | `WorkspaceView` | Protected | Projects, real analytics, Kanban tasks, notes, **Reading Room**, **RAG SSE Streaming Copilot** |
| `/collections` | `CollectionsView` | Protected | Unified library, AI overview, **K-Means+TF-IDF Topic Clusters**, **Cross-Collection Insights** |
| `/saved` | `SavedPapersView` | Protected | All-saved library + **Research Debt Tracker** |
| `/network` | `NetworkView` | Protected | Researcher directory + **Collaboration Affinity Score** badges |
| `/trending` | `TrendingView` | Protected | Emerging topic momentum & clusters |
| `/publish` | `PublishView` | Protected | Manuscript submission, PDF upload, AI pre-review, **Citation Gap Detector** |
| `/notifications` | `NotificationsView` | Protected | Activity center |
| `/settings` | `SettingsView` | Protected | Profile editor, verification, **Research Velocity Dashboard** |

## 2. State, Sync, & Error Visibility (`lib/platform-store.ts`)

- **Backend Status**: Tracks `backendStatus` (`'connecting' | 'live' | 'demo'`), `syncError`, and `writeError`.
- **Visible Connection Banner**: `ClientShell.tsx` renders an explicit *"Using demo data: Backend API unreachable"* banner with a **Retry** button when offline, and a dismissible error banner whenever a write fails.
- **Auth Cookies**: `lib/api/client.ts` sets `ckgai_auth=1` on login/register/dev-login and `ckgai_logged_out=1` on explicit logout so Next.js `middleware.ts` enforces session state cleanly.

## 3. Typed API Client (`lib/api/`)

All backend calls go through `lib/api/client.ts` (JWT Bearer header + automatic 401 refresh retry):
- `auth.ts`, `papers.ts`, `search.ts`, `researchers.ts`, `collections.ts`, `feed.ts`, `notifications.ts`, `projects.ts`, `manuscripts.ts`, `graph.ts`, `insights.ts`.
- Streaming RAG chat connects directly to `/api/v1/projects/{id}/ai-thread/stream` (with Next.js `/api/chat-assistant/stream` proxy fallback).
