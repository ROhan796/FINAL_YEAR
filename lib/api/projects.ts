import { api } from './client';

export interface Member {
  name: string;
  role: 'Owner' | 'Editor' | 'Viewer';
  email: string;
  avatar: string;
}

export interface Task {
  id: string;
  title: string;
  assigneeName: string;
  assigneeAvatar: string;
  dueDate: string;
  status: 'Todo' | 'In Progress' | 'Done';
}

export interface Note {
  id: string;
  title: string;
  content: string;
  paperId?: string;
  paperTitle?: string;
  updatedAt: string;
  authorName: string;
}

export interface Dataset {
  id: string;
  name: string;
  description: string;
  size: string;
  format: string;
  link: string;
  uploadedDate: string;
}

export interface Discussion {
  id: string;
  authorName: string;
  authorAvatar: string;
  content: string;
  timestamp: string;
}

export interface Document {
  id: string;
  title: string;
  status: string;
  updatedAt: string;
  aiScore?: number;
}

export interface ActivityPoint {
  date: string;
  papersAdded: number;
  notesCreated: number;
  tasksCompleted: number;
}

export interface WorkspaceProject {
  id: string;
  name: string;
  description: string;
  field: string;
  status: 'Active' | 'Under Review' | 'Archived';
  ownerName: string;
  members: Member[];
  paperIds: string[];
  notes: Note[];
  tasks: Task[];
  milestones: Record<string, unknown>[];
  datasets: Dataset[];
  documents: Document[];
  discussions: Discussion[];
  activityHistory: ActivityPoint[];
}

export interface PageResponse<T> {
  items: T[];
  total: number;
  nextCursor?: string | null;
}

export const projectsApi = {
  list: () => api.get<PageResponse<WorkspaceProject>>('/api/v1/projects'),

  get: (id: string) => api.get<WorkspaceProject>(`/api/v1/projects/${id}`),

  create: (data: { name: string; description?: string; field?: string }) =>
    api.post<WorkspaceProject>('/api/v1/projects', data),

  delete: (id: string) => api.delete<{ ok: boolean }>(`/api/v1/projects/${id}`),

  addTask: (projectId: string, data: { title: string; dueDate?: string; assigneeName?: string }) =>
    api.post<Task>(`/api/v1/projects/${projectId}/tasks`, data),

  updateTask: (projectId: string, taskId: string, data: { status?: string; title?: string }) =>
    api.patch<Task>(`/api/v1/projects/${projectId}/tasks/${taskId}`, data),

  deleteTask: (projectId: string, taskId: string) =>
    api.delete<{ ok: boolean }>(`/api/v1/projects/${projectId}/tasks/${taskId}`),

  addNote: (projectId: string, data: { title: string; content: string; paperId?: string }) =>
    api.post<Note>(`/api/v1/projects/${projectId}/notes`, data),

  updateNote: (projectId: string, noteId: string, data: { title?: string; content?: string }) =>
    api.patch<Note>(`/api/v1/projects/${projectId}/notes/${noteId}`, data),

  deleteNote: (projectId: string, noteId: string) =>
    api.delete<{ ok: boolean }>(`/api/v1/projects/${projectId}/notes/${noteId}`),

  addDiscussion: (projectId: string, content: string) =>
    api.post<Discussion>(`/api/v1/projects/${projectId}/discussions`, { content }),

  deleteDiscussion: (projectId: string, msgId: string) =>
    api.delete<{ ok: boolean }>(`/api/v1/projects/${projectId}/discussions/${msgId}`),

  addPaper: (projectId: string, paperId: string) =>
    api.post<{ ok: boolean }>(`/api/v1/projects/${projectId}/papers`, { paperId }),

  removePaper: (projectId: string, paperId: string) =>
    api.delete<{ ok: boolean }>(`/api/v1/projects/${projectId}/papers/${paperId}`),

  inviteMember: (projectId: string, email: string, role: string = 'Viewer') =>
    api.post<Member>(`/api/v1/projects/${projectId}/members`, { email, role }),

  removeMember: (projectId: string, email: string) =>
    api.delete<{ ok: boolean }>(`/api/v1/projects/${projectId}/members/${encodeURIComponent(email)}`),

  addDataset: (projectId: string, data: { name: string; description?: string; format?: string; size?: string; link?: string }) =>
    api.post<Dataset>(`/api/v1/projects/${projectId}/datasets`, data),

  deleteDataset: (projectId: string, datasetId: string) =>
    api.delete<{ ok: boolean }>(`/api/v1/projects/${projectId}/datasets/${datasetId}`),

  getAnalytics: (projectId: string) =>
    api.get<{ totalHours: number; papersAdded: number; notesCreated: number; tasksCompleted: number; tasksTotal: number; chart: ActivityPoint[] }>(
      `/api/v1/projects/${projectId}/analytics`,
    ),

  aiThread: (projectId: string, message: string) =>
    api.post<{ reply: string }>(`/api/v1/projects/${projectId}/ai-thread`, { message }),
};
