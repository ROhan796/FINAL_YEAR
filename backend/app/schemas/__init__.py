"""Pydantic wire contracts.

Field names deliberately mirror `lib/mock-data.ts` (camelCase: `openAccess`,
`citationCount`, `isLiked`, ...) so the Next.js client can consume responses
without a translation layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.schemas.common import (
    ErrorBody,
    HealthResponse,
    MetaResponse,
    OkResponse,
    PageResponse,
)

PostType = Literal[
    "Research Update",
    "Preprint Announcement",
    "Methodology Discussion",
    "Dataset Release",
]
RelevanceSignal = Literal["High", "Medium", "Low"]
TaskStatus = Literal["Todo", "In Progress", "Done"]
ProjectStatus = Literal["Active", "Under Review", "Archived"]
MemberRole = Literal["Owner", "Editor", "Viewer"]
ManuscriptStatus = Literal[
    "Draft",
    "Under AI Review",
    "Submitted",
    "Under Review",
    "Published",
    "Revision Required",
]
NotificationType = Literal[
    "mention", "comment", "collaboration", "paper_update", "verification"
]
SearchMode = Literal["semantic", "keyword", "hybrid"]
CitationFormat = Literal["APA", "MLA", "BibTeX", "RIS"]


# ── users ────────────────────────────────────────────────────────────────────


class UserProfileOut(BaseModel):
    """`UserProfile` from the frontend store."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    name: str
    title: str
    institution: str
    field: str
    bio: str
    avatar: str
    verified: bool
    interests: list[str] = []
    email: str


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    name: str = Field(min_length=1, max_length=200)
    title: str = ""
    institution: str = ""
    field: str = ""


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenPair(BaseModel):
    accessToken: str
    refreshToken: str
    user: UserProfileOut


class ProfileUpdate(BaseModel):
    name: str | None = None
    title: str | None = None
    institution: str | None = None
    field: str | None = None
    bio: str | None = None
    avatar: str | None = None
    interests: list[str] | None = None


class VerificationRequestIn(BaseModel):
    institutionalEmail: EmailStr
    orcid: str = Field(min_length=3, max_length=64)


class VerificationRequestOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    status: str
    institutionalEmail: str = Field(validation_alias="institutional_email")
    orcid: str
    createdAt: datetime = Field(validation_alias="created_at")
    decidedAt: datetime | None = Field(default=None, validation_alias="decided_at")


# ── papers ───────────────────────────────────────────────────────────────────


class PaperOut(BaseModel):
    """`Paper` from the frontend — every optional AI field preserved."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    title: str
    authors: list[str] = []
    abstract: str = ""
    year: int
    venue: str = ""
    doi: str = ""
    tags: list[str] = []
    citationCount: int = Field(0, validation_alias="citation_count")
    openAccess: bool = Field(False, validation_alias="open_access")
    pdfUrl: str | None = Field(default=None, validation_alias="pdf_url")

    relevanceSignal: RelevanceSignal | None = Field(
        default=None, validation_alias="relevanceSignal"
    )
    reasoningEvidence: str | None = Field(default=None, validation_alias="reasoningEvidence")

    aiSummary: str | None = Field(default=None, validation_alias="ai_summary")
    keyContributions: list[str] | None = Field(default=None, validation_alias="key_contributions")
    methodology: str | None = None
    limitations: list[str] | None = None
    researchGaps: list[str] | None = Field(default=None, validation_alias="research_gaps")
    citationContext: str | None = Field(default=None, validation_alias="citation_context")


class PaperFilters(BaseModel):
    q: str | None = None
    tag: str | None = None
    year: int | None = None
    minCitations: int | None = None
    openAccess: bool | None = None
    sort: Literal["relevance", "citations", "newest", "year"] = "relevance"


class CitationOut(BaseModel):
    format: CitationFormat
    citation: str


# ── search ───────────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str = Field(min_length=0, max_length=1000)
    mode: SearchMode = "hybrid"
    filters: PaperFilters | None = None
    cursor: str | None = None
    limit: int = Field(default=20, ge=1, le=100)
    explain: bool = False


class SearchResponse(BaseModel):
    results: list[PaperOut]
    query: str
    mode: SearchMode
    queryUnderstanding: str = ""
    latencyMs: int
    suggestedSearches: list[str] = []
    total: int = 0
    nextCursor: str | None = None
    arms: list[str] = []
    isHttpAccelerated: bool = False


# ── feed ─────────────────────────────────────────────────────────────────────


class FeedCommentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    authorName: str = Field(validation_alias="author_name")
    authorInstitution: str = Field(validation_alias="author_institution")
    authorAvatar: str = Field(validation_alias="author_avatar")
    content: str
    timestamp: str
    verified: bool = Field(validation_alias="author_verified")


class FeedPostOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    authorId: str = Field(validation_alias="author_id")
    authorName: str = Field(validation_alias="author_name")
    authorTitle: str = Field(validation_alias="author_title")
    authorInstitution: str = Field(validation_alias="author_institution")
    authorAvatar: str = Field(validation_alias="author_avatar")
    authorVerified: bool = Field(validation_alias="author_verified")
    content: str
    postType: PostType = Field(validation_alias="post_type")
    timestamp: str
    likes: int = Field(validation_alias="like_count")
    linkedPaper: PaperOut | None = Field(default=None, validation_alias="linked_paper")
    comments: list[FeedCommentOut] = []
    tags: list[str] = []
    isLiked: bool = False


class PostCreate(BaseModel):
    content: str = Field(min_length=1, max_length=10000)
    postType: PostType = "Research Update"
    linkedPaperId: str | None = None


class CommentCreate(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class PostLikeOut(OkResponse):
    """Like/unlike ack. Extends `OkResponse` with the viewer's new state so the
    client can flip the heart without refetching the whole post."""

    isLiked: bool = False
    likes: int = 0


# ── researchers ──────────────────────────────────────────────────────────────


class ResearcherOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    name: str
    title: str
    institution: str
    field: str
    bio: str
    avatar: str
    verified: bool
    publicationsCount: int = Field(validation_alias="publications_count")
    citationCount: int = Field(validation_alias="citation_count")
    hIndex: int = Field(validation_alias="h_index")
    followersCount: int = Field(validation_alias="follower_count")
    interests: list[str] = []
    publications: list[PaperOut] = []
    isFollowed: bool = False


# ── collections ──────────────────────────────────────────────────────────────


class CollectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    name: str
    description: str
    pinned: bool
    paperIds: list[str] = Field(validation_alias="paper_ids")
    createdDate: str = Field(validation_alias="created_date")
    aiSummary: str | None = Field(default=None, validation_alias="ai_summary")
    shareableLink: str | None = None


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = ""
    paperId: str | None = None


class CollectionPatch(BaseModel):
    name: str | None = None
    description: str | None = None
    pinned: bool | None = None


class CollectionPaperIn(BaseModel):
    paperId: str


class AiOverviewOut(BaseModel):
    summary: str
    # CollectionsView historically read `executiveSummary`; return both so the
    # dead feature works without a frontend change.
    executiveSummary: str
    paperCount: int = 0


# ── workspace ────────────────────────────────────────────────────────────────


class MemberOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    name: str
    role: MemberRole
    email: str
    avatar: str


class TaskOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    title: str
    assigneeName: str = Field(validation_alias="assignee_name")
    assigneeAvatar: str = Field(validation_alias="assignee_avatar")
    dueDate: str
    status: TaskStatus


class NoteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    title: str
    content: str
    paperId: str | None = Field(default=None, validation_alias="paper_id")
    paperTitle: str | None = Field(default=None, validation_alias="paper_title")
    updatedAt: str
    authorName: str = Field(validation_alias="author_name")


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    name: str
    description: str
    size: str
    format: str
    link: str
    uploadedDate: str = Field(validation_alias="created_at")


class DiscussionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    authorName: str = Field(validation_alias="author_name")
    authorAvatar: str = Field(validation_alias="author_avatar")
    content: str
    timestamp: str


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    title: str
    status: str
    updatedAt: str
    aiScore: int | None = Field(default=None, validation_alias="ai_score")


class ActivityPoint(BaseModel):
    date: str
    papersAdded: int = 0
    notesCreated: int = 0
    tasksCompleted: int = 0


class WorkspaceProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    name: str
    description: str
    field: str
    status: ProjectStatus
    ownerName: str = Field(validation_alias="owner_name")
    members: list[MemberOut] = []
    paperIds: list[str] = Field(validation_alias="paper_ids")
    notes: list[NoteOut] = []
    tasks: list[TaskOut] = []
    milestones: list[dict[str, Any]] = []
    datasets: list[DatasetOut] = []
    documents: list[DocumentOut] = []
    discussions: list[DiscussionOut] = []
    activityHistory: list[ActivityPoint] = []


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = ""
    field: str = ""


class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    dueDate: str | None = None
    assigneeName: str | None = None


class TaskPatch(BaseModel):
    status: TaskStatus | None = None
    title: str | None = None
    dueDate: str | None = None


class NoteCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1)
    paperId: str | None = None


class DiscussionCreate(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class MemberInvite(BaseModel):
    email: EmailStr
    role: MemberRole = "Viewer"


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=300)
    description: str = ""
    format: str = ""
    size: str = ""
    link: str = ""


class ChatTurn(BaseModel):
    sender: Literal["user", "assistant"]
    text: str


class ChatReply(BaseModel):
    reply: str
    cached: bool = False


class ProjectAnalytics(BaseModel):
    totalHours: float = 0.0
    papersAdded: int = 0
    notesCreated: int = 0
    tasksCompleted: int = 0
    tasksTotal: int = 0
    chart: list[ActivityPoint] = []


# ── publishing ───────────────────────────────────────────────────────────────


class AIReviewOut(BaseModel):
    predictedScore: int
    titleReview: str
    executiveCritique: str
    suggestedKeywords: list[str]
    likelyPeerQuestions: list[str]
    clarityAssessment: str
    strengths: list[str]
    weaknesses: list[str]


class ManuscriptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    title: str
    abstract: str
    authors: list[str] = []
    coAuthors: list[str] = Field(validation_alias="co_authors")
    venue: str
    tags: list[str] = []
    year: int
    fileName: str = Field(validation_alias="file_name")
    fileSize: str = Field(validation_alias="file_size")
    integrityConfirmed: bool = Field(validation_alias="integrity_confirmed")
    status: ManuscriptStatus
    submittedAt: str = Field(validation_alias="submitted_at")
    aiReview: AIReviewOut | None = Field(default=None, validation_alias="ai_review")


class ManuscriptCreate(BaseModel):
    title: str = Field(min_length=1)
    abstract: str = ""
    authors: list[str] = []
    coAuthors: list[str] = []
    venue: str = ""
    tags: list[str] = []
    year: int = 2026
    integrityConfirmed: bool = False


class ManuscriptStatusPatch(BaseModel):
    status: ManuscriptStatus


# ── notifications / trending ────────────────────────────────────────────────


class NotificationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    type: NotificationType
    title: str
    message: str
    timestamp: str
    read: bool
    link: str | None = None


class UnreadCountOut(BaseModel):
    """Navbar badge payload."""

    count: int = 0


class TrendingTopicOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    name: str
    category: str
    monthlyGrowth: float = Field(validation_alias="monthly_growth")
    citationCount: int = Field(validation_alias="citation_count")
    paperCount: int = Field(validation_alias="paper_count")
    relatedPaperIds: list[str] = Field(validation_alias="related_paper_ids")
    topResearchers: list[str] = Field(validation_alias="top_researchers")
    aiNarrative: str = Field(validation_alias="ai_narrative")
    clusters: list[str] = []


# ── graph / discovery ────────────────────────────────────────────────────────


class GraphNodeOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    label: str
    entityType: str = Field(validation_alias="entity_type")
    domain: str
    degree: int = 0


class GraphEdgeOut(BaseModel):
    id: str
    source: str
    target: str
    relation: str
    confidence: float


class GraphExplorationOut(BaseModel):
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]
    centerId: str | None = None
    truncated: bool = False


class DiscoveryOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    score: float
    relation: str
    srcNode: GraphNodeOut
    dstNode: GraphNodeOut
    evidencePaths: list[list[str]] = Field(default_factory=list, validation_alias="evidence_paths")
    evidencePaperIds: list[str] = Field(default_factory=list, validation_alias="evidence_paper_ids")
    crossDomain: bool = True


class DiscoveryFeedback(BaseModel):
    useful: bool
