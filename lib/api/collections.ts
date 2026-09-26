import { api } from './client';
import type { Paper } from './papers';

export interface Collection {
  id: string;
  name: string;
  description: string;
  pinned: boolean;
  paperIds: string[];
  createdDate: string;
  aiSummary?: string;
  shareableLink?: string;
}

export interface CollectionPaper {
  paperId: string;
}

export interface PageResponse<T> {
  items: T[];
  total: number;
  nextCursor?: string | null;
}

export const collectionsApi = {
  list: () => api.get<PageResponse<Collection>>('/api/v1/collections'),

  get: (id: string) => api.get<Collection>(`/api/v1/collections/${id}`),

  create: (data: { name: string; description?: string; paperId?: string }) =>
    api.post<Collection>('/api/v1/collections', data),

  update: (id: string, data: { name?: string; description?: string; pinned?: boolean }) =>
    api.patch<Collection>(`/api/v1/collections/${id}`, data),

  delete: (id: string) => api.delete<{ ok: boolean }>(`/api/v1/collections/${id}`),

  addPaper: (collectionId: string, paperId: string) =>
    api.post<{ ok: boolean }>(`/api/v1/collections/${collectionId}/papers`, { paperId }),

  removePaper: (collectionId: string, paperId: string) =>
    api.delete<{ ok: boolean }>(`/api/v1/collections/${collectionId}/papers/${paperId}`),

  aiOverview: (collectionId: string) =>
    api.post<{ summary: string; executiveSummary: string; paperCount: number }>(
      `/api/v1/collections/${collectionId}/ai-overview`,
    ),

  export: (collectionId: string) =>
    api.get<string>(`/api/v1/collections/${collectionId}/export`),
};
