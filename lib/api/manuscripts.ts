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

  uploadFile: (id: string, _file: File) => {
    // File upload — backend expects multipart, but for now POST metadata
    return api.post<{ ok: boolean }>(`/api/v1/manuscripts/${id}/file`, {});
  },

  aiPreReview: (id: string) =>
    api.post<AIReview>(`/api/v1/manuscripts/${id}/ai-prereview`),
};
