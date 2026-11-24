"""Ingestion adapters — arXiv, PubMed, Semantic Scholar."""

from ml.ingest.arxiv import ARXIV_API, IngestedPaper, fetch_arxiv, fetch_arxiv_by_ids
from ml.ingest.pubmed import EFETCH_URL, ESEARCH_URL, fetch_pubmed
from ml.ingest.semantic_scholar import S2_SEARCH_URL, fetch_semantic_scholar
from ml.pipelines.ingest import IngestPipeline, IngestResult

__all__ = [
    "ARXIV_API",
    "EFETCH_URL",
    "ESEARCH_URL",
    "IngestPipeline",
    "IngestResult",
    "IngestedPaper",
    "S2_SEARCH_URL",
    "fetch_arxiv",
    "fetch_arxiv_by_ids",
    "fetch_pubmed",
    "fetch_semantic_scholar",
]
