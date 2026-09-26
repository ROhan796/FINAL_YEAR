import { api } from './client';
import type { Paper } from './papers';

export interface FeedComment {
  id: string;
  authorName: string;
  authorInstitution: string;
  authorAvatar: string;
  content: string;
  timestamp: string;
  verified: boolean;
}

export interface FeedPost {
  id: string;
  authorId: string;
  authorName: string;
  authorTitle: string;
  authorInstitution: string;
  authorAvatar: string;
  authorVerified: boolean;
  content: string;
  postType: 'Research Update' | 'Preprint Announcement' | 'Methodology Discussion' | 'Dataset Release';
  timestamp: string;
  likes: number;
  linkedPaper?: Paper;
  comments: FeedComment[];
  tags: string[];
  isLiked: boolean;
}

export interface PageResponse<T> {
  items: T[];
  total: number;
  nextCursor?: string | null;
}

export const feedApi = {
  getFeed: (cursor?: string) =>
    api.get<PageResponse<FeedPost>>('/api/v1/feed', {
      params: { cursor },
    }),

  getPost: (id: string) => api.get<FeedPost>(`/api/v1/posts/${id}`),

  createPost: (data: { content: string; postType?: string; linkedPaperId?: string }) =>
    api.post<FeedPost>('/api/v1/posts', data),

  addComment: (postId: string, content: string) =>
    api.post<FeedComment[]>(`/api/v1/posts/${postId}/comments`, { content }),

  toggleLike: (postId: string) =>
    api.post<{ ok: boolean; isLiked: boolean; likes: number }>(`/api/v1/posts/${postId}/like`),
};
