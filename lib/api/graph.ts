import { api } from './client';

export interface GraphNode {
  id: string;
  label: string;
  entityType: string;
  domain: string;
  degree: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  confidence: number;
}

export interface GraphExploration {
  nodes: GraphNode[];
  edges: GraphEdge[];
  centerId?: string;
  truncated: boolean;
}

export interface Discovery {
  id: string;
  score: number;
  relation: string;
  srcNode: GraphNode;
  dstNode: GraphNode;
  evidencePaths: string[][];
  evidencePaperIds: string[];
  crossDomain: boolean;
}

export const graphApi = {
  exploration: (nodeId?: string, hops: number = 2, relation?: string) =>
    api.get<GraphExploration>('/api/v1/graph/exploration', {
      params: { nodeId, hops, relation },
    }),

  getDiscoveries: (limit: number = 50) =>
    api.get<Discovery[]>('/api/v1/discoveries', { params: { limit } }),

  feedback: (discoveryId: string, useful: boolean) =>
    api.post<Discovery>(`/api/v1/discoveries/${discoveryId}/feedback`, { useful }),

  retrain: (trainGnn: boolean = true, epochs: number = 25) =>
    api.post<{
      modelTrained: boolean;
      discoveriesCount: number;
      persisted: number;
      hitsAt10: number;
      mrr: number;
    }>('/api/v1/discoveries/retrain', undefined, { params: { trainGnn, epochs } }),

  projectionStatus: () =>
    api.get<{
      neo4jEnabled: boolean;
      lastProjectedAt: string | null;
      nodesProjected: number;
      edgesProjected: number;
      error: string | null;
    }>('/api/v1/graph/projection-status'),

  projectToNeo4j: (full: boolean = false) =>
    api.post<{ status: string; nodesProjected: number; edgesProjected: number; error: string | null }>(
      '/api/v1/graph/project',
      undefined,
      { params: { full } },
    ),
};
