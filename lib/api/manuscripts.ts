import { api } from './client';

export interface AIReview {
  predictedScore: number;
  titleReview: string;
  executiveCritique: string;
  suggestedKeywords: string[];
  likelyPeerQuestions: string[];
  clarityAssessment: string;
  strengths: string[];
  weaknesses: string[];
}

export interface Manuscript {
  id: string;
  title: string;
  abstract: string;
  authors: string[];
  coAuthors: string[];
  venue: string;
  tags: string[];
  year: number;
  fileName: string;
  fileSize: string;
  integrityConfirmed: boolean;
  status: 'Draft' | 'Under AI Review' | 'Submitted' | 'Under Review' | 'Published' | 'Revision Required';
  submittedAt: string;
  aiReview?: AIReview;
}

export interface PageResponse<T> {
  items: T[];
  total: number;
  nextCursor?: string | null;
}

export const manuscriptsApi = {
  list: () => api.get<PageResponse<Manuscript>>('/api/v1/manuscripts'),

  get: (id: string) => api.get<Manuscript>(`/api/v1/manuscripts/${id}`),

  create: (data: {
    title: string;
    abstract?: string;
    authors?: string[];
    coAuthors?: string[];
    venue?: string;
    tags?: string[];
    year?: number;
    integrityConfirmed?: boolean;
  }) => api.post<Manuscript>('/api/v1/manuscripts', data),

  updateStatus: (id: string, status: string) =>
    api.patch<Manuscript>(`/api/v1/manuscripts/${id}/status`, { status }),

  uploadFile: async (id: string, file: File) => {
    const form = new FormData();
    form.append('file', file);
    const { API_BASE, getAccessToken } = await import('./client');
    const headers: Record<string, string> = {};
    const tok = getAccessToken();
    if (tok) headers['Authorization'] = `Bearer ${tok}`;
    const res = await fetch(`${API_BASE}/api/v1/manuscripts/${id}/file`, {
      method: 'POST',
      headers,
      body: form,
    });
    if (!res.ok) throw new Error(`File upload failed: ${res.status}`);
    return res.json() as Promise<Manuscript>;
  },

  aiPreReview: (id: string) =>
    api.post<{ aiReview: AIReview; manuscriptId: string }>(`/api/v1/manuscripts/${id}/ai-prereview`),
};
