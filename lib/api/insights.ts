import { api } from './client';

export interface ResearchGap {
  id: string;
  method: string;
  methodDomain: string;
  target: string;
  targetType: string;
  targetDomain: string;
  gapScore: number;
  summary: string;
  supportingPaperIds: string[];
}

export interface GatewayPaper {
  rank: number;
  paperId: string;
  title: string;
  year: number;
  venue: string;
  domain: string;
  citationCount: number;
  gatewayScore: number;
  inDegree: number;
  rationale: string;
}

export interface ResearchDebtItem {
  paperId: string;
  title: string;
  venue: string;
  year: number;
  savedAt: string;
  ageDays: number;
  isOverdue: boolean;
  missing: string[];
  debtReason: string;
}

export interface ResearchDebtResponse {
  totalSaved: number;
  debtCount: number;
  overdueCount: number;
  items: ResearchDebtItem[];
}

export interface CrossCollectionInsight {
  collectionA: { id: string; name: string };
  collectionB: { id: string; name: string };
  headline: string;
  bridgingPapers: {
    id: string;
    title: string;
    year: number;
    venue: string;
    bridgeScore: number;
  }[];
}

export interface ResearcherAffinity {
  researcherId: string;
  affinityScore: number;
  tier: 'High' | 'Medium' | 'Emerging';
  sharedTopics: string[];
  reason: string;
}

export interface VelocityResponse {
  totals: {
    papersSaved: number;
    collections: number;
    notes: number;
    tasksDone: number;
    manuscripts: number;
  };
  series: {
    month: string;
    papersSaved: number;
    collectionsCreated: number;
    notesWritten: number;
    tasksDone: number;
    manuscriptsSubmitted: number;
  }[];
}

export interface CollectionCluster {
  clusterId: number;
  label: string;
  paperIds: string[];
  paperTitles: string[];
  topTerms: string[];
}

export interface PaperExplainability {
  paperId: string;
  title: string;
  reasons: { source: string; title: string; detail: string }[];
}

export interface CitationPolarityOut {
  paperId: string;
  citationSentence: string;
  label: 'supports' | 'disputes' | 'neutral';
  confidence: number;
  probabilities: Record<string, number>;
  evidence: string;
  modelUsed: string;
}

export interface CitationGapsOut {
  manuscriptId: string;
  gapCount: number;
  gaps: {
    entity: string;
    entityType: string;
    message: string;
    candidates: {
      id: string;
      title: string;
      year: number;
      venue: string;
      citationCount: number;
    }[];
  }[];
}

export interface ReadingAnnotation {
  id: string;
  projectId: string;
  paperId: string;
  chunkId?: string | null;
  sentenceIndex: number;
  highlightText: string;
  comment: string;
  authorName: string;
  authorAvatar: string;
  timestamp: string;
}

export interface IngestStatus {
  totalPapers: number;
  embeddedPapers: number;
  paperChunks: number;
  kgNodes: number;
  kgEdges: number;
  discoveries: number;
  sources: Record<string, number>;
  adapters: { arxiv: boolean; pubmed: boolean; semanticScholar: boolean };
  queueEnabled: boolean;
}

export const insightsApi = {
  researchGaps: (limit: number = 15) =>
    api.get<ResearchGap[]>('/api/v1/insights/research-gaps', { params: { limit } }),

  gatewayPapers: (limit: number = 8) =>
    api.get<GatewayPaper[]>('/api/v1/insights/gateway-papers', { params: { limit } }),

  researchDebt: (agingDays: number = 30) =>
    api.get<ResearchDebtResponse>('/api/v1/insights/research-debt', { params: { agingDays } }),

  crossCollection: (limit: number = 6) =>
    api.get<CrossCollectionInsight[]>('/api/v1/insights/cross-collection', { params: { limit } }),

  affinity: () =>
    api.get<Record<string, ResearcherAffinity>>('/api/v1/insights/affinity'),

  velocity: () =>
    api.get<VelocityResponse>('/api/v1/insights/velocity'),

  collectionClusters: (collectionId: string) =>
    api.get<{ collectionId: string; clusters: CollectionCluster[] }>(
      `/api/v1/collections/${collectionId}/clusters`,
    ),

  explainPaper: (paperId: string, q: string = '') =>
    api.get<PaperExplainability>(`/api/v1/search/explain/${paperId}`, { params: { q } }),

  citationPolarity: (paperId: string, citationSentence: string = '') =>
    api.post<CitationPolarityOut>(`/api/v1/papers/${paperId}/citation-polarity`, undefined, {
      params: { citationSentence },
    }),

  manuscriptCitationGaps: (manuscriptId: string, draftText: string = '') =>
    api.post<CitationGapsOut>(`/api/v1/manuscripts/${manuscriptId}/citation-gaps`, undefined, {
      params: { draftText },
    }),

  readingAnnotations: (projectId: string, paperId?: string) =>
    api.get<ReadingAnnotation[]>(`/api/v1/projects/${projectId}/reading-room`, {
      params: { paperId },
    }),

  addReadingAnnotation: (
    projectId: string,
    data: { paperId: string; sentenceIndex: number; highlightText: string; comment: string; chunkId?: string },
  ) => api.post<ReadingAnnotation>(`/api/v1/projects/${projectId}/reading-room`, data),

  deleteReadingAnnotation: (projectId: string, annotationId: string) =>
    api.delete<{ ok: boolean }>(`/api/v1/projects/${projectId}/reading-room/${annotationId}`),

  ingestStatus: () => api.get<IngestStatus>('/api/v1/ingest/status'),

  triggerIngest: (source: 'arxiv' | 'pubmed' | 'semantic-scholar', query: string, maxResults: number = 15) =>
    api.post<{ queued: boolean; jobId: string | null; source: string; backend: string }>(
      `/api/v1/ingest/${source}`,
      { query, maxResults, embed: true, buildKg: true },
    ),
};
