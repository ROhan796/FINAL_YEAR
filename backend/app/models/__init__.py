"""ORM model registry.

Importing this module registers every table on `Base.metadata` — Alembic and
`create_all` both depend on it.
"""

from app.models.collection import Collection, CollectionPaper
from app.models.knowledge import Discovery, GraphProjectionState, KgEdge, KgNode
from app.models.manuscript import Manuscript
from app.models.notification import Notification, TrendingTopic
from app.models.paper import Paper, PaperChunk, PaperTag, RelatedPaper, SavedPaper, SearchHistory
from app.models.project import (
    Project,
    ProjectChatMessage,
    ProjectDataset,
    ProjectDiscussion,
    ProjectDocument,
    ProjectMember,
    ProjectNote,
    ProjectPaper,
    ProjectTask,
    ProjectTimeEntry,
    ReadingAnnotation,
)
from app.models.social import (
    FeedComment,
    FeedPost,
    Follow,
    PostLike,
    ResearcherProfile,
)
from app.models.user import User, UserProfile, VerificationRequest

__all__ = [
    "Collection",
    "CollectionPaper",
    "Discovery",
    "FeedComment",
    "FeedPost",
    "Follow",
    "GraphProjectionState",
    "KgEdge",
    "KgNode",
    "Manuscript",
    "Notification",
    "Paper",
    "PaperChunk",
    "PaperTag",
    "PostLike",
    "Project",
    "ProjectChatMessage",
    "ProjectDataset",
    "ProjectDiscussion",
    "ProjectDocument",
    "ProjectMember",
    "ProjectNote",
    "ProjectPaper",
    "ProjectTask",
    "ProjectTimeEntry",
    "ReadingAnnotation",
    "RelatedPaper",
    "ResearcherProfile",
    "SavedPaper",
    "SearchHistory",
    "TrendingTopic",
    "User",
    "UserProfile",
    "VerificationRequest",
]
