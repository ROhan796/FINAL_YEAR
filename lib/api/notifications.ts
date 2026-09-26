import { api } from './client';

export interface Notification {
  id: string;
  type: 'mention' | 'comment' | 'collaboration' | 'paper_update' | 'verification';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  link?: string;
}

export interface PageResponse<T> {
  items: T[];
  total: number;
  nextCursor?: string | null;
}

export const notificationsApi = {
  list: () => api.get<PageResponse<Notification>>('/api/v1/notifications'),

  getUnreadCount: () => api.get<{ count: number }>('/api/v1/notifications/unread-count'),

  markRead: (id: string) =>
    api.patch<Notification>(`/api/v1/notifications/${id}`, { read: true }),

  markAllRead: () => api.post<{ ok: boolean }>('/api/v1/notifications/read-all'),
};
