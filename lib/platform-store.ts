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
import { authApi, papersApi, feedApi, collectionsApi, researchersApi, projectsApi, manuscriptsApi, notificationsApi, searchApi } from './api';

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
}

function getInitialState(): PlatformState {
  if (typeof window === 'undefined') {
    return {
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
      searchHistory: ['multimodal reasoning', 'quantum surface spectroscopy', 'homomorphic encryption genomics']
    };
  }

  try {
    const item = window.localStorage.getItem(STORAGE_KEY);
    if (item) {
      const parsed = JSON.parse(item);
      return {
        user: parsed.user || DEFAULT_USER,
        savedPaperIds: parsed.savedPaperIds || ['paper-101', 'paper-104'],
        papers: parsed.papers || INITIAL_PAPERS,
        researchers: parsed.researchers || INITIAL_RESEARCHERS,
        feedPosts: parsed.feedPosts || INITIAL_FEED_POSTS,
        collections: parsed.collections || INITIAL_COLLECTIONS,
        projects: parsed.projects || INITIAL_PROJECTS,
        notifications: parsed.notifications || INITIAL_NOTIFICATIONS,
        trendingTopics: parsed.trendingTopics || INITIAL_TRENDING_TOPICS,
        manuscripts: parsed.manuscripts || INITIAL_MANUSCRIPTS,
        searchHistory: parsed.searchHistory || ['multimodal reasoning', 'quantum surface spectroscopy']
      };
    }
  } catch (e) {
    console.error('Error loading state from localStorage:', e);
  }

  return {
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
    searchHistory: ['multimodal reasoning', 'quantum surface spectroscopy']
  };
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
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(nextState));
    } catch (e) {
      console.error('Error saving state to localStorage:', e);
    }
  }
  listeners.forEach((listener) => listener(nextState));
}

/** Partial state update — merges into current state without replacing everything. */
function patchStore(patch: Partial<PlatformState>) {
  setStoreState((prev) => ({ ...prev, ...patch }));
}

// ── Backend sync (runs once on mount) ──────────────────────────────────

async function syncFromBackend() {
  if (backendSynced || typeof window === 'undefined') return;
  backendSynced = true;

  try {
    // 1. Auth: dev-login to get tokens + user
    const tokens = await authApi.devLogin();
    const user: UserProfile = tokens.user;
    patchStore({ user });

    // 2. Papers
    try {
      const paperRes = await papersApi.list();
      if (paperRes.items?.length) {
        patchStore({ papers: paperRes.items as unknown as Paper[] });
      }
    } catch { /* keep mock */ }

    // 3. Saved papers
    try {
      const savedRes = await papersApi.getSaved();
      if (savedRes.items) {
        patchStore({ savedPaperIds: savedRes.items.map((p: Paper) => p.id) });
      }
    } catch { /* keep mock */ }

    // 4. Feed
    try {
      const feedRes = await feedApi.getFeed();
      if (feedRes.items?.length) {
        patchStore({ feedPosts: feedRes.items as unknown as FeedPost[] });
      }
    } catch { /* keep mock */ }

    // 5. Collections
    try {
      const colRes = await collectionsApi.list();
      if (colRes.items?.length) {
        patchStore({ collections: colRes.items as unknown as Collection[] });
      }
    } catch { /* keep mock */ }

    // 6. Researchers
    try {
      const resRes = await researchersApi.list();
      if (resRes.items?.length) {
        patchStore({ researchers: resRes.items as unknown as Researcher[] });
      }
    } catch { /* keep mock */ }

    // 7. Projects
    try {
      const projRes = await projectsApi.list();
      if (projRes.items?.length) {
        patchStore({ projects: projRes.items as unknown as WorkspaceProject[] });
      }
    } catch { /* keep mock */ }

    // 8. Manuscripts
    try {
      const msRes = await manuscriptsApi.list();
      if (msRes.items?.length) {
        patchStore({ manuscripts: msRes.items as unknown as ManuscriptSubmission[] });
      }
    } catch { /* keep mock */ }

    // 9. Notifications
    try {
      const notifRes = await notificationsApi.list();
      if (notifRes.items?.length) {
        patchStore({ notifications: notifRes.items as unknown as NotificationItem[] });
      }
    } catch { /* keep mock */ }

    // 10. Search history
    try {
      const histRes = await searchApi.getHistory();
      if (histRes?.length) {
        patchStore({ searchHistory: histRes });
      }
    } catch { /* keep mock */ }

  } catch (err) {
    console.warn('Backend sync failed, using local data:', err);
  }
}

export function usePlatformStore() {
  const [state, setState] = useState<PlatformState>(getStoreState);

  useEffect(() => {
    const listener = (newState: PlatformState) => {
      setState(newState);
    };
    listeners.add(listener);

    // Trigger backend sync on first mount
    syncFromBackend();

    return () => {
      listeners.delete(listener);
    };
  }, []);

  const toggleSavePaper = useCallback((paperId: string) => {
    // Optimistic local update
    setStoreState((prev) => {
      const isSaved = prev.savedPaperIds.includes(paperId);
      const updatedSaved = isSaved
        ? prev.savedPaperIds.filter((id) => id !== paperId)
        : [...prev.savedPaperIds, paperId];
      return { ...prev, savedPaperIds: updatedSaved };
    });
    // Backend call (fire and forget)
    papersApi.save(paperId).catch(() => papersApi.unsave(paperId).catch(() => {}));
  }, []);

  const toggleFollowResearcher = useCallback((researcherId: string) => {
    setStoreState((prev) => {
      const updated = prev.researchers.map((r) => {
        if (r.id === researcherId) {
          const isFollowed = !r.isFollowed;
          return {
            ...r,
            isFollowed,
            followersCount: isFollowed ? r.followersCount + 1 : Math.max(0, r.followersCount - 1)
          };
        }
        return r;
      });
      return { ...prev, researchers: updated };
    });
    researchersApi.follow(researcherId).catch(() => researchersApi.unfollow(researcherId).catch(() => {}));
  }, []);

  const toggleLikePost = useCallback((postId: string) => {
    setStoreState((prev) => {
      const updatedPosts = prev.feedPosts.map((post) => {
        if (post.id === postId) {
          const isLiked = !post.isLiked;
          return {
            ...post,
            isLiked,
            likes: isLiked ? post.likes + 1 : Math.max(0, post.likes - 1)
          };
        }
        return post;
      });
      return { ...prev, feedPosts: updatedPosts };
    });
    feedApi.toggleLike(postId).catch(() => {});
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
      const updatedPosts = prev.feedPosts.map((post) => {
        if (post.id === postId) {
          return { ...post, comments: [...post.comments, newComment] };
        }
        return post;
      });
      return { ...prev, feedPosts: updatedPosts };
    });
    feedApi.addComment(postId, commentText).catch(() => {});
  }, []);

  const createPost = useCallback((content: string, postType: FeedPost['postType'], linkedPaper?: Paper) => {
    if (!content.trim()) return;
    feedApi.createPost({ content, postType, linkedPaperId: linkedPaper?.id }).then((newPost) => {
      setStoreState((prev) => ({
        ...prev,
        feedPosts: [newPost as unknown as FeedPost, ...prev.feedPosts]
      }));
    }).catch(() => {
      // Fallback: local-only post
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
    }).catch(() => {
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
      if (col) collectionsApi.update(collectionId, { pinned: !col.pinned }).catch(() => {});
      const updated = prev.collections.map((c) => (c.id === collectionId ? { ...c, pinned: !c.pinned } : c));
      return { ...prev, collections: updated };
    });
  }, []);

  const addPaperToCollection = useCallback((collectionId: string, paperId: string) => {
    setStoreState((prev) => {
      const updated = prev.collections.map((c) => {
        if (c.id === collectionId && !c.paperIds.includes(paperId)) {
          collectionsApi.addPaper(collectionId, paperId).catch(() => {});
          return { ...c, paperIds: [...c.paperIds, paperId] };
        }
        return c;
      });
      return { ...prev, collections: updated };
    });
  }, []);

  const removePaperFromCollection = useCallback((collectionId: string, paperId: string) => {
    setStoreState((prev) => {
      const updated = prev.collections.map((c) => {
        if (c.id === collectionId) {
          collectionsApi.removePaper(collectionId, paperId).catch(() => {});
          return { ...c, paperIds: c.paperIds.filter((id) => id !== paperId) };
        }
        return c;
      });
      return { ...prev, collections: updated };
    });
  }, []);

  const deleteCollection = useCallback((collectionId: string) => {
    collectionsApi.delete(collectionId).catch(() => {});
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
    }).catch(() => {
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
      setStoreState((prev) => {
        const updatedProjects = prev.projects.map((p) => {
          if (p.id === projectId) return { ...p, tasks: [...p.tasks, newTask as unknown as Task] };
          return p;
        });
        return { ...prev, projects: updatedProjects };
      });
    }).catch(() => {
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
        const updatedProjects = prev.projects.map((p) => {
          if (p.id === projectId) return { ...p, tasks: [...p.tasks, newTask] };
          return p;
        });
        return { ...prev, projects: updatedProjects };
      });
    });
  }, []);

  const updateTaskStatus = useCallback((projectId: string, taskId: string, newStatus: Task['status']) => {
    setStoreState((prev) => {
      const updatedProjects = prev.projects.map((p) => {
        if (p.id === projectId) {
          projectsApi.updateTask(projectId, taskId, { status: newStatus }).catch(() => {});
          const updatedTasks = p.tasks.map((t) => (t.id === taskId ? { ...t, status: newStatus } : t));
          return { ...p, tasks: updatedTasks };
        }
        return p;
      });
      return { ...prev, projects: updatedProjects };
    });
  }, []);

  const addNoteToProject = useCallback((projectId: string, title: string, content: string, paperId?: string) => {
    projectsApi.addNote(projectId, { title, content, paperId }).then((newNote) => {
      setStoreState((prev) => {
        const updatedProjects = prev.projects.map((p) => {
          if (p.id === projectId) return { ...p, notes: [newNote as unknown as import('./mock-data').ProjectNote, ...p.notes] };
          return p;
        });
        return { ...prev, projects: updatedProjects };
      });
    }).catch(() => {
      const currentUser = getStoreState().user;
      const papers = getStoreState().papers;
      const paper = papers.find((p) => p.id === paperId);
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
        const updatedProjects = prev.projects.map((p) => {
          if (p.id === projectId) return { ...p, notes: [newNote, ...p.notes] };
          return p;
        });
        return { ...prev, projects: updatedProjects };
      });
    });
  }, []);

  const addProjectDiscussion = useCallback((projectId: string, content: string) => {
    if (!content.trim()) return;
    projectsApi.addDiscussion(projectId, content).then((newMsg) => {
      setStoreState((prev) => {
        const updatedProjects = prev.projects.map((p) => {
          if (p.id === projectId) return { ...p, discussions: [...p.discussions, newMsg as unknown as import('./mock-data').ProjectDiscussion] };
          return p;
        });
        return { ...prev, projects: updatedProjects };
      });
    }).catch(() => {
      const currentUser = getStoreState().user;
      setStoreState((prev) => {
        const newMsg = {
          id: 'disc-' + Date.now(),
          authorName: currentUser.name,
          authorAvatar: currentUser.avatar,
          content: content.trim(),
          timestamp: 'Just now'
        };
        const updatedProjects = prev.projects.map((p) => {
          if (p.id === projectId) return { ...p, discussions: [...p.discussions, newMsg] };
          return p;
        });
        return { ...prev, projects: updatedProjects };
      });
    });
  }, []);

  const inviteCollaborator = useCallback((projectId: string, email: string, role: 'Owner' | 'Editor' | 'Viewer') => {
    projectsApi.inviteMember(projectId, email, role).then((newMember) => {
      setStoreState((prev) => {
        const updatedProjects = prev.projects.map((p) => {
          if (p.id === projectId) return { ...p, members: [...p.members, newMember as unknown as import('./mock-data').WorkspaceProject['members'][0]] };
          return p;
        });
        return { ...prev, projects: updatedProjects };
      });
    }).catch(() => {
      const nameFromEmail = email.split('@')[0].replace('.', ' ');
      setStoreState((prev) => {
        const newMember = {
          name: nameFromEmail.charAt(0).toUpperCase() + nameFromEmail.slice(1),
          role,
          email,
          avatar: `https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=400&auto=format&fit=crop&q=80`
        };
        const updatedProjects = prev.projects.map((p) => {
          if (p.id === projectId) return { ...p, members: [...p.members, newMember] };
          return p;
        });
        return { ...prev, projects: updatedProjects };
      });
    });
  }, []);

  const markNotificationRead = useCallback((id: string) => {
    notificationsApi.markRead(id).catch(() => {});
    setStoreState((prev) => ({
      ...prev,
      notifications: prev.notifications.map((n) => (n.id === id ? { ...n, read: true } : n))
    }));
  }, []);

  const markAllNotificationsRead = useCallback(() => {
    notificationsApi.markAllRead().catch(() => {});
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
    }).catch(() => {
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
    }).catch(() => {
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
