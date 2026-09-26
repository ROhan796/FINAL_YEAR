import { api } from './client';
import type { Paper, PaperFilters } from './papers';

export interface SearchResponse {
  results: Paper[];
  query: string;
  mode: string;
  queryUnderstanding: string;
  latencyMs: number;
  suggestedSearches: string[];
  total: number;
  nextCursor?: string | null;
  arms: string[];
  isHttpAccelerated: boolean;
}

export const searchApi = {
  search: (query: string, options?: {
    mode?: 'semantic' | 'keyword' | 'hybrid';
    filters?: PaperFilters;
    limit?: number;
    isHttpAccelerated?: boolean;
  }) =>
    api.post<SearchResponse>('/api/v1/search', {
      query,
      mode: options?.mode || 'hybrid',
      filters: options?.filters,
      limit: options?.limit || 20,
      explain: true,
    }),

  getHistory: () => api.get<string[]>('/api/v1/search/history'),

  addHistory: (query: string) =>
    api.post<string[]>('/api/v1/search/history', { query }),

  clearHistory: () => api.delete<string[]>('/api/v1/search/history'),

  getSuggestions: (limit: number = 6) =>
    api.get<string[]>('/api/v1/search/suggestions', { params: { limit } }),
};
