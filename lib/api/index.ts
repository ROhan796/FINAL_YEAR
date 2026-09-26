export { api, setTokens, clearTokens, getAccessToken, getRefreshToken, ApiError, API_BASE } from './client';
export type { ApiError as ApiErrorType } from './client';

export { authApi } from './auth';
export type { UserProfile, TokenPair, AuthConfig } from './auth';

export { papersApi } from './papers';
export type { Paper, PaperFilters, CitationOut, PageResponse as PaperPageResponse } from './papers';

export { searchApi } from './search';
export type { SearchResponse } from './search';

export { researchersApi } from './researchers';
export type { Researcher } from './researchers';

export { collectionsApi } from './collections';
export type { Collection } from './collections';

export { feedApi } from './feed';
export type { FeedPost, FeedComment } from './feed';

export { notificationsApi } from './notifications';
export type { Notification } from './notifications';

export { projectsApi } from './projects';
export type { WorkspaceProject, Task, Note, Member, Dataset, Discussion, Document, ActivityPoint } from './projects';

export { manuscriptsApi } from './manuscripts';
export type { Manuscript, AIReview } from './manuscripts';

export { graphApi } from './graph';
export type { GraphNode, GraphEdge, GraphExploration, Discovery } from './graph';
