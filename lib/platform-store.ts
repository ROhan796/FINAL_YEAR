import { useState, useEffect, useCallback } from 'react';
import {
  Paper,
  Researcher,
  FeedPost,
  Collection,
  Task,
  WorkspaceProject,
  NotificationItem,
  TrendingTopic,
  ManuscriptSubmission,
  INITIAL_PAPERS,
  INITIAL_RESEARCHERS,
  INITIAL_FEED_POSTS,
  INITIAL_COLLECTIONS,
  INITIAL_PROJECTS,
  INITIAL_NOTIFICATIONS,
  INITIAL_TRENDING_TOPICS,
  INITIAL_MANUSCRIPTS
} from './mock-data';
import {
  api,
  authApi,
  papersApi,
  feedApi,
  collectionsApi,
  researchersApi,
  projectsApi,
  manuscriptsApi,
  notificationsApi,
  searchApi,
  getAccessToken,
} from './api';

export interface UserProfile {
  id: string;
  name: string;
  title: string;
  institution: string;
  field: string;
  bio: string;
  avatar: string;
  verified: boolean;
  interests: string[];
  email: string;
}

export type BackendStatus = 'connecting' | 'live' | 'demo';

const DEFAULT_USER: UserProfile = {
  id: 'current-user',
  name: 'Dr. Amara Nwosu',
  title: 'Principal AI Researcher',
  institution: 'Independent Research Institute',
  field: 'Computational Linguistics & AI Reasoning',
  bio: 'Independent researcher focusing on multimodal reasoning architectures, adaptive transformer routing, and causal physics engines.',
  avatar: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400&auto=format&fit=crop&q=80',
  verified: true,
  interests: ['Multimodal Transformers', 'Causal AI', 'Symbolic Reasoning', 'Open Weights'],
  email: 'amara.nwosu@research-inst.org'
};

const STORAGE_KEY = 'ai_native_research_platform_state_v1';

interface PlatformState {
  user: UserProfile;
  savedPaperIds: string[];
  papers: Paper[];
  researchers: Researcher[];
  feedPosts: FeedPost[];
  collections: Collection[];
  projects: WorkspaceProject[];
  notifications: NotificationItem[];
  trendingTopics: TrendingTopic[];
  manuscripts: ManuscriptSubmission[];
  searchHistory: string[];
  backendStatus: BackendStatus;
  syncError: string | null;
  writeError: string | null;
}

function getInitialState(): PlatformState {
  const base: PlatformState = {
    user: DEFAULT_USER,
    savedPaperIds: ['paper-101', 'paper-104'],
    papers: INITIAL_PAPERS,
    researchers: INITIAL_RESEARCHERS,
    feedPosts: INITIAL_FEED_POSTS,
    collections: INITIAL_COLLECTIONS,
    projects: INITIAL_PROJECTS,
    notifications: INITIAL_NOTIFICATIONS,
    trendingTopics: INITIAL_TRENDING_TOPICS,
    manuscripts: INITIAL_MANUSCRIPTS,
    searchHistory: ['multimodal reasoning', 'quantum surface spectroscopy', 'homomorphic encryption genomics'],
    backendStatus: 'connecting',
    syncError: null,
    writeError: null,
  };

  if (typeof window === 'undefined') {
    return base;
  }

  try {
    const item = window.localStorage.getItem(STORAGE_KEY);
    if (item) {
      const parsed = JSON.parse(item);
      return {
        ...base,
        user: parsed.user || DEFAULT_USER,
        savedPaperIds: parsed.savedPaperIds || base.savedPaperIds,
        papers: parsed.papers || INITIAL_PAPERS,
        researchers: parsed.researchers || INITIAL_RESEARCHERS,
        feedPosts: parsed.feedPosts || INITIAL_FEED_POSTS,
        collections: parsed.collections || INITIAL_COLLECTIONS,
        projects: parsed.projects || INITIAL_PROJECTS,
        notifications: parsed.notifications || INITIAL_NOTIFICATIONS,
        trendingTopics: parsed.trendingTopics || INITIAL_TRENDING_TOPICS,
        manuscripts: parsed.manuscripts || INITIAL_MANUSCRIPTS,
        searchHistory: parsed.searchHistory || base.searchHistory,
      };
    }
  } catch (e) {
    console.error('Error loading state from localStorage:', e);
  }

  return base;
}

let globalState: PlatformState | null = null;
let isInitializedOnClient = false;
let backendSynced = false;
const listeners = new Set<(state: PlatformState) => void>();

function getStoreState(): PlatformState {
  if (typeof window !== 'undefined' && !isInitializedOnClient) {
    isInitializedOnClient = true;
    globalState = getInitialState();
  } else if (!globalState) {
    globalState = getInitialState();
  }
  return globalState;
}

function setStoreState(updater: (prev: PlatformState) => PlatformState) {
  const nextState = updater(getStoreState());
  globalState = nextState;
  if (typeof window !== 'undefined') {
    try {
      // Persist domain entities only (keep transient status fresh across reloads)
      const { backendStatus: _b, syncError: _s, writeError: _w, ...persistable } = nextState;
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(persistable));
    } catch (e) {
      console.error('Error saving state to localStorage:', e);
    }
  }
  listeners.forEach((listener) => listener(nextState));
}

function patchStore(patch: Partial<PlatformState>) {
  setStoreState((prev) => ({ ...prev, ...patch }));
}

function reportWriteError(action: string, err: unknown) {
  const msg = err instanceof Error ? err.message : 'Backend request failed';
  patchStore({ writeError: `${action}: ${msg}` });
}

// ── Backend sync ──────────────────────────────────────────────────────

async function syncFromBackend(force: boolean = false) {
  if (typeof window === 'undefined') return;
  if (backendSynced && !force) return;
  backendSynced = true;
  patchStore({ backendStatus: 'connecting', syncError: null });

  try {
    const loggedOut = document.cookie.includes('ckgai_logged_out=1');
    let user: UserProfile | null = null;

    if (getAccessToken()) {
      try {
        user = await authApi.getMe();
      } catch {
        user = null;
      }
    }

    if (!user && !loggedOut) {
      const tokens = await authApi.devLogin();
      user = tokens.user;
    }

    if (user) {
      patchStore({ user });
    }

    const results = await Promise.allSettled([
      papersApi.list({ sort: 'citations' }),
      papersApi.getSaved(),
      feedApi.getFeed(),
      collectionsApi.list(),
      researchersApi.list(),
      projectsApi.list(),
      manuscriptsApi.list(),
      notificationsApi.list(),
      searchApi.getHistory(),
      api.get<{ items: TrendingTopic[] }>('/api/v1/trending'),
    ]);

    const [
      paperRes,
      savedRes,
      feedRes,
      colRes,
      resRes,
      projRes,
      msRes,
      notifRes,
      histRes,
      trendRes,
    ] = results;

    const patch: Partial<PlatformState> = { backendStatus: 'live', syncError: null };

    if (paperRes.status === 'fulfilled' && paperRes.value.items?.length) {
      patch.papers = paperRes.value.items as unknown as Paper[];
    }
    if (savedRes.status === 'fulfilled' && savedRes.value.items) {
      patch.savedPaperIds = savedRes.value.items.map((p: Paper) => p.id);
    }
    if (feedRes.status === 'fulfilled' && feedRes.value.items?.length) {
      patch.feedPosts = feedRes.value.items as unknown as FeedPost[];
    }
    if (colRes.status === 'fulfilled' && colRes.value.items?.length) {
      patch.collections = colRes.value.items as unknown as Collection[];
    }
    if (resRes.status === 'fulfilled' && resRes.value.items?.length) {
      patch.researchers = resRes.value.items as unknown as Researcher[];
    }
    if (projRes.status === 'fulfilled' && projRes.value.items?.length) {
      patch.projects = projRes.value.items as unknown as WorkspaceProject[];
    }
    if (msRes.status === 'fulfilled' && Array.isArray(msRes.value)) {
      patch.manuscripts = msRes.value as unknown as ManuscriptSubmission[];
    } else if (msRes.status === 'fulfilled' && (msRes.value as { items?: ManuscriptSubmission[] }).items) {
      patch.manuscripts = (msRes.value as { items: ManuscriptSubmission[] }).items;
    }
    if (notifRes.status === 'fulfilled' && notifRes.value.items) {
      patch.notifications = notifRes.value.items as unknown as NotificationItem[];
    }
    if (histRes.status === 'fulfilled' && histRes.value?.length) {
      patch.searchHistory = histRes.value;
    }
    if (trendRes.status === 'fulfilled' && trendRes.value.items?.length) {
      patch.trendingTopics = trendRes.value.items;
    }

    patchStore(patch);
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Backend API unreachable';
    patchStore({ backendStatus: 'demo', syncError: msg });
  }
}

export function usePlatformStore() {
  const [state, setState] = useState<PlatformState>(getStoreState);

  useEffect(() => {
    const listener = (newState: PlatformState) => {
      setState(newState);
    };
    listeners.add(listener);
    syncFromBackend(false);
    return () => {
      listeners.delete(listener);
    };
  }, []);

  const retrySync = useCallback(async () => {
    await syncFromBackend(true);
  }, []);

  const clearWriteError = useCallback(() => {
    patchStore({ writeError: null });
  }, []);

  const logout = useCallback(() => {
    authApi.logout();
    backendSynced = false;
    patchStore({ backendStatus: 'demo' });
  }, []);

  const toggleSavePaper = useCallback((paperId: string) => {
    const wasSaved = getStoreState().savedPaperIds.includes(paperId);
    setStoreState((prev) => ({
      ...prev,
      savedPaperIds: wasSaved
        ? prev.savedPaperIds.filter((id) => id !== paperId)
        : [...prev.savedPaperIds, paperId]
    }));
    const req = wasSaved ? papersApi.unsave(paperId) : papersApi.save(paperId);
    req.catch((err) => reportWriteError('Save paper sync failed', err));
  }, []);

  const toggleFollowResearcher = useCallback((researcherId: string) => {
    const target = getStoreState().researchers.find((r) => r.id === researcherId);
    const willFollow = !(target?.isFollowed);
    setStoreState((prev) => ({
      ...prev,
      researchers: prev.researchers.map((r) =>
        r.id === researcherId
          ? {
              ...r,
              isFollowed: willFollow,
              followersCount: willFollow ? r.followersCount + 1 : Math.max(0, r.followersCount - 1)
            }
          : r
      )
    }));
    const req = willFollow ? researchersApi.follow(researcherId) : researchersApi.unfollow(researcherId);
    req.catch((err) => reportWriteError('Follow update failed', err));
  }, []);

  const toggleLikePost = useCallback((postId: string) => {
    setStoreState((prev) => ({
      ...prev,
      feedPosts: prev.feedPosts.map((post) => {
        if (post.id === postId) {
          const isLiked = !post.isLiked;
          return {
            ...post,
            isLiked,
            likes: isLiked ? post.likes + 1 : Math.max(0, post.likes - 1)
          };
        }
        return post;
      })
    }));
    feedApi.toggleLike(postId).catch((err) => reportWriteError('Like sync failed', err));
  }, []);

  const addCommentToPost = useCallback((postId: string, commentText: string) => {
    if (!commentText.trim()) return;
    const currentUser = getStoreState().user;
    setStoreState((prev) => {
      const newComment = {
        id: 'c-' + Date.now(),
        authorName: currentUser.name,
        authorInstitution: currentUser.institution,
        authorAvatar: currentUser.avatar,
        content: commentText.trim(),
        timestamp: 'Just now',
        verified: currentUser.verified
      };
      return {
        ...prev,
        feedPosts: prev.feedPosts.map((post) =>
          post.id === postId ? { ...post, comments: [...post.comments, newComment] } : post
        )
      };
    });
    feedApi.addComment(postId, commentText).catch((err) => reportWriteError('Comment sync failed', err));
  }, []);

  const createPost = useCallback((content: string, postType: FeedPost['postType'], linkedPaper?: Paper) => {
    if (!content.trim()) return;
    feedApi.createPost({ content, postType, linkedPaperId: linkedPaper?.id }).then((newPost) => {
      setStoreState((prev) => ({
        ...prev,
        feedPosts: [newPost as unknown as FeedPost, ...prev.feedPosts]
      }));
    }).catch((err) => {
      reportWriteError('Post saved locally (backend offline)', err);
      const currentUser = getStoreState().user;
      setStoreState((prev) => {
        const newPost: FeedPost = {
          id: 'post-' + Date.now(),
          authorId: currentUser.id,
          authorName: currentUser.name,
          authorTitle: currentUser.title,
          authorInstitution: currentUser.institution,
          authorAvatar: currentUser.avatar,
          authorVerified: currentUser.verified,
          content: content.trim(),
          postType,
          timestamp: 'Just now',
          likes: 0,
          linkedPaper,
          comments: [],
          tags: [postType, 'Research'],
          isLiked: false
        };
        return { ...prev, feedPosts: [newPost, ...prev.feedPosts] };
      });
    });
  }, []);

  const createCollection = useCallback((name: string, description: string) => {
    collectionsApi.create({ name, description }).then((newCol) => {
      setStoreState((prev) => ({
        ...prev,
        collections: [newCol as unknown as Collection, ...prev.collections]
      }));
    }).catch((err) => {
      reportWriteError('Collection created locally (backend sync failed)', err);
      setStoreState((prev) => {
        const newCol: Collection = {
          id: 'col-' + Date.now(),
          name,
          description,
          pinned: false,
          paperIds: [],
          createdDate: new Date().toISOString().split('T')[0],
          aiSummary: 'Collection created.',
          shareableLink: `https://ai.studio/collections/col-${Date.now()}?share=token`
        };
        return { ...prev, collections: [newCol, ...prev.collections] };
      });
    });
  }, []);

  const togglePinCollection = useCallback((collectionId: string) => {
    setStoreState((prev) => {
      const col = prev.collections.find((c) => c.id === collectionId);
      if (col) {
        collectionsApi.update(collectionId, { pinned: !col.pinned }).catch((err) =>
          reportWriteError('Pin update failed', err)
        );
      }
      return {
        ...prev,
        collections: prev.collections.map((c) => (c.id === collectionId ? { ...c, pinned: !c.pinned } : c))
      };
    });
  }, []);

  const addPaperToCollection = useCallback((collectionId: string, paperId: string) => {
    setStoreState((prev) => ({
      ...prev,
      collections: prev.collections.map((c) => {
        if (c.id === collectionId && !c.paperIds.includes(paperId)) {
          collectionsApi.addPaper(collectionId, paperId).catch((err) =>
            reportWriteError('Add paper to collection failed', err)
          );
          return { ...c, paperIds: [...c.paperIds, paperId] };
        }
        return c;
      })
    }));
  }, []);

  const removePaperFromCollection = useCallback((collectionId: string, paperId: string) => {
    setStoreState((prev) => ({
      ...prev,
      collections: prev.collections.map((c) => {
        if (c.id === collectionId) {
          collectionsApi.removePaper(collectionId, paperId).catch((err) =>
            reportWriteError('Remove paper failed', err)
          );
          return { ...c, paperIds: c.paperIds.filter((id) => id !== paperId) };
        }
        return c;
      })
    }));
  }, []);

  const deleteCollection = useCallback((collectionId: string) => {
    collectionsApi.delete(collectionId).catch((err) => reportWriteError('Delete collection failed', err));
    setStoreState((prev) => ({
      ...prev,
      collections: prev.collections.filter((c) => c.id !== collectionId)
    }));
  }, []);

  const createWorkspaceProject = useCallback((name: string, description: string, field: string) => {
    projectsApi.create({ name, description, field }).then((newProj) => {
      setStoreState((prev) => ({
        ...prev,
        projects: [newProj as unknown as WorkspaceProject, ...prev.projects]
      }));
    }).catch((err) => {
      reportWriteError('Project created locally', err);
      const currentUser = getStoreState().user;
      setStoreState((prev) => {
        const newProj: WorkspaceProject = {
          id: 'proj-' + Date.now(),
          name,
          description,
          field,
          status: 'Active',
          ownerName: currentUser.name,
          members: [{ name: currentUser.name, role: 'Owner', email: currentUser.email, avatar: currentUser.avatar }],
          paperIds: [],
          notes: [],
          tasks: [],
          milestones: [],
          datasets: [],
          documents: [],
          discussions: [],
          activityHistory: [{ date: 'Mon', papersAdded: 1, notesCreated: 1, tasksCompleted: 0 }]
        };
        return { ...prev, projects: [newProj, ...prev.projects] };
      });
    });
  }, []);

  const addTaskToProject = useCallback((projectId: string, title: string, dueDate: string, assigneeName: string) => {
    projectsApi.addTask(projectId, { title, dueDate, assigneeName }).then((newTask) => {
      setStoreState((prev) => ({
        ...prev,
        projects: prev.projects.map((p) =>
          p.id === projectId ? { ...p, tasks: [...p.tasks, newTask as unknown as Task] } : p
        )
      }));
    }).catch((err) => {
      reportWriteError('Task sync failed', err);
      const currentUser = getStoreState().user;
      setStoreState((prev) => {
        const newTask = {
          id: 't-' + Date.now(),
          title,
          dueDate: dueDate || '2026-08-30',
          assigneeName: assigneeName || currentUser.name,
          assigneeAvatar: currentUser.avatar,
          status: 'Todo' as const
        };
        return {
          ...prev,
          projects: prev.projects.map((p) =>
            p.id === projectId ? { ...p, tasks: [...p.tasks, newTask] } : p
          )
        };
      });
    });
  }, []);

  const updateTaskStatus = useCallback((projectId: string, taskId: string, newStatus: Task['status']) => {
    setStoreState((prev) => ({
      ...prev,
      projects: prev.projects.map((p) => {
        if (p.id === projectId) {
          projectsApi.updateTask(projectId, taskId, { status: newStatus }).catch((err) =>
            reportWriteError('Task status update failed', err)
          );
          return {
            ...p,
            tasks: p.tasks.map((t) => (t.id === taskId ? { ...t, status: newStatus } : t))
          };
        }
        return p;
      })
    }));
  }, []);

  const addNoteToProject = useCallback((projectId: string, title: string, content: string, paperId?: string) => {
    projectsApi.addNote(projectId, { title, content, paperId }).then((newNote) => {
      setStoreState((prev) => ({
        ...prev,
        projects: prev.projects.map((p) =>
          p.id === projectId
            ? { ...p, notes: [newNote as unknown as import('./mock-data').ProjectNote, ...p.notes] }
            : p
        )
      }));
    }).catch((err) => {
      reportWriteError('Note sync failed', err);
      const currentUser = getStoreState().user;
      const paper = getStoreState().papers.find((p) => p.id === paperId);
      setStoreState((prev) => {
        const newNote = {
          id: 'note-' + Date.now(),
          title,
          content,
          paperId,
          paperTitle: paper?.title,
          updatedAt: new Date().toISOString().split('T')[0],
          authorName: currentUser.name
        };
        return {
          ...prev,
          projects: prev.projects.map((p) =>
            p.id === projectId ? { ...p, notes: [newNote, ...p.notes] } : p
          )
        };
      });
    });
  }, []);

  const addProjectDiscussion = useCallback((projectId: string, content: string) => {
    if (!content.trim()) return;
    projectsApi.addDiscussion(projectId, content).then((newMsg) => {
      setStoreState((prev) => ({
        ...prev,
        projects: prev.projects.map((p) =>
          p.id === projectId
            ? { ...p, discussions: [...p.discussions, newMsg as unknown as import('./mock-data').ProjectDiscussion] }
            : p
        )
      }));
    }).catch((err) => {
      reportWriteError('Discussion sync failed', err);
      const currentUser = getStoreState().user;
      setStoreState((prev) => {
        const newMsg = {
          id: 'disc-' + Date.now(),
          authorName: currentUser.name,
          authorAvatar: currentUser.avatar,
          content: content.trim(),
          timestamp: 'Just now'
        };
        return {
          ...prev,
          projects: prev.projects.map((p) =>
            p.id === projectId ? { ...p, discussions: [...p.discussions, newMsg] } : p
          )
        };
      });
    });
  }, []);

  const inviteCollaborator = useCallback((projectId: string, email: string, role: 'Owner' | 'Editor' | 'Viewer') => {
    projectsApi.inviteMember(projectId, email, role).then((proj) => {
      const members = (proj as unknown as WorkspaceProject).members;
      setStoreState((prev) => ({
        ...prev,
        projects: prev.projects.map((p) =>
          p.id === projectId
            ? { ...p, members: Array.isArray(members) ? members : [...p.members, proj as unknown as WorkspaceProject['members'][0]] }
            : p
        )
      }));
    }).catch((err) => {
      reportWriteError('Member invite sync failed', err);
      const nameFromEmail = email.split('@')[0].replace('.', ' ');
      setStoreState((prev) => {
        const newMember = {
          name: nameFromEmail.charAt(0).toUpperCase() + nameFromEmail.slice(1),
          role,
          email,
          avatar: 'https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=400&auto=format&fit=crop&q=80'
        };
        return {
          ...prev,
          projects: prev.projects.map((p) =>
            p.id === projectId ? { ...p, members: [...p.members, newMember] } : p
          )
        };
      });
    });
  }, []);

  const markNotificationRead = useCallback((id: string) => {
    notificationsApi.markRead(id).catch((err) => reportWriteError('Notification update failed', err));
    setStoreState((prev) => ({
      ...prev,
      notifications: prev.notifications.map((n) => (n.id === id ? { ...n, read: true } : n))
    }));
  }, []);

  const markAllNotificationsRead = useCallback(() => {
    notificationsApi.markAllRead().catch((err) => reportWriteError('Notifications update failed', err));
    setStoreState((prev) => ({
      ...prev,
      notifications: prev.notifications.map((n) => ({ ...n, read: true }))
    }));
  }, []);

  const toggleVerificationUser = useCallback(() => {
    setStoreState((prev) => ({
      ...prev,
      user: { ...prev.user, verified: !prev.user.verified }
    }));
  }, []);

  const updateProfile = useCallback((updatedProfile: Partial<UserProfile>) => {
    authApi.updateProfile(updatedProfile).then((profile) => {
      setStoreState((prev) => ({ ...prev, user: profile as unknown as UserProfile }));
    }).catch((err) => {
      reportWriteError('Profile saved locally', err);
      setStoreState((prev) => ({ ...prev, user: { ...prev.user, ...updatedProfile } }));
    });
  }, []);

  const addSearchHistory = useCallback((query: string) => {
    if (!query.trim()) return;
    searchApi.addHistory(query).catch(() => {});
    setStoreState((prev) => {
      const filtered = prev.searchHistory.filter((q) => q.toLowerCase() !== query.toLowerCase());
      return { ...prev, searchHistory: [query.trim(), ...filtered].slice(0, 10) };
    });
  }, []);

  const submitManuscript = useCallback((manuscript: ManuscriptSubmission) => {
    manuscriptsApi.create({
      title: manuscript.title,
      abstract: manuscript.abstract,
      authors: manuscript.authors,
      coAuthors: manuscript.coAuthors,
      venue: manuscript.venue,
      tags: manuscript.tags,
      year: manuscript.year,
      integrityConfirmed: manuscript.integrityConfirmed,
    }).then((newMs) => {
      setStoreState((prev) => ({
        ...prev,
        manuscripts: [newMs as unknown as ManuscriptSubmission, ...prev.manuscripts]
      }));
    }).catch((err) => {
      reportWriteError('Manuscript saved locally', err);
      setStoreState((prev) => ({
        ...prev,
        manuscripts: [manuscript, ...prev.manuscripts]
      }));
    });
  }, []);

  return {
    state,
    user: state.user,
    savedPaperIds: state.savedPaperIds,
    papers: state.papers,
    researchers: state.researchers,
    feedPosts: state.feedPosts,
    collections: state.collections,
    projects: state.projects,
    notifications: state.notifications,
    trendingTopics: state.trendingTopics,
    manuscripts: state.manuscripts,
    searchHistory: state.searchHistory,
    backendStatus: state.backendStatus,
    syncError: state.syncError,
    writeError: state.writeError,
    retrySync,
    clearWriteError,
    logout,
    toggleSavePaper,
    toggleFollowResearcher,
    toggleLikePost,
    addCommentToPost,
    createPost,
    createCollection,
    togglePinCollection,
    addPaperToCollection,
    removePaperFromCollection,
    deleteCollection,
    createWorkspaceProject,
    addTaskToProject,
    updateTaskStatus,
    addNoteToProject,
    addProjectDiscussion,
    inviteCollaborator,
    markNotificationRead,
    markAllNotificationsRead,
    toggleVerificationUser,
    updateProfile,
    addSearchHistory,
    submitManuscript
  };
}
