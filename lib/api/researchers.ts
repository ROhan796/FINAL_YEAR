import { api } from './client';
import type { Paper } from './papers';

export interface Researcher {
  id: string;
  name: string;
  title: string;
  institution: string;
  field: string;
  bio: string;
  avatar: string;
  verified: boolean;
  publicationsCount: number;
  citationCount: number;
  hIndex: number;
  followersCount: number;
  interests: string[];
  publications: Paper[];
  isFollowed: boolean;
}

export interface PageResponse<T> {
  items: T[];
  total: number;
  nextCursor?: string | null;
}

export const researchersApi = {
  list: () => api.get<PageResponse<Researcher>>('/api/v1/researchers'),

  get: (id: string) => api.get<Researcher>(`/api/v1/researchers/${id}`),

  follow: (id: string) =>
    api.post<{ ok: boolean; isFollowed: boolean }>(`/api/v1/researchers/${id}/follow`),

  unfollow: (id: string) =>
    api.delete<{ ok: boolean; isFollowed: boolean }>(`/api/v1/researchers/${id}/follow`),

  getFollowers: (id: string) =>
    api.get<PageResponse<Researcher>>(`/api/v1/researchers/${id}/followers`),

  getFollowing: (id: string) =>
    api.get<PageResponse<Researcher>>(`/api/v1/researchers/${id}/following`),
};
