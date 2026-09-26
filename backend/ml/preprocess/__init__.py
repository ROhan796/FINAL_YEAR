"""Text preprocessing — normalize, segment, tokenize, section detection.

Layer 2 of the CKGAI pipeline.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Strip HTML, LaTeX, normalize Unicode."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove LaTeX math
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)
    # Remove figure/table references
    text = re.sub(r"\(Fig\.?\s*\d+\)", "", text)
    text = re.sub(r"\(Table\s*\d+\)", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def segment_sentences(text: str) -> list[str]:
    """Split text into sentences (handles scientific abbreviations)."""
    # Handle common scientific abbreviations that shouldn't split
    abbrevs = ["Fig", "fig", "et al", "i.e", "e.g", "etc", "Dr", "Mr", "Mrs",
               "vs", "al", "approx", "eq", "ref", "vol", "no", "pp"]
    for abbr in abbrevs:
        text = text.replace(f"{abbr}.", f"{abbr}[[PERIOD]]")

    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Restore periods
    sentences = [s.replace("[[PERIOD]]", ".") for s in sentences]
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def identify_sections(text: str) -> dict[str, str]:
    """Identify structural sections of a paper."""
    section_patterns = {
        "abstract": r"(?i)\babstract\b[:\s]*(.+?)(?=\b(?:introduction|keywords)\b)",
        "introduction": r"(?i)\bintroduction\b[:\s]*(.+?)(?=\b(?:methods|methodology|related work|background)\b)",
        "methods": r"(?i)\b(?:methods?|methodology)\b[:\s]*(.+?)(?=\b(?:results?|experiments?|evaluation)\b)",
        "results": r"(?i)\b(?:results?|experiments?|evaluation)\b[:\s]*(.+?)(?=\b(?:discussion|conclusion)\b)",
        "conclusion": r"(?i)\bconclusions?\b[:\s]*(.+?)$",
    }
    sections = {}
    for name, pattern in section_patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[name] = match.group(1).strip()
    return sections


def tokenize_for_transformer(text: str, max_length: int = 512) -> list[str]:
    """Basic tokenization for transformer input (placeholder for real tokenizer)."""
    tokens = text.split()
    return tokens[:max_length]


def preprocess_paper(title: str, abstract: str) -> dict:
    """Full preprocessing pipeline for a paper."""
    full_text = f"{title}. {abstract}"
    normalized = normalize_text(full_text)
    sentences = segment_sentences(normalized)
    sections = identify_sections(normalized)

    return {
        "title": normalize_text(title),
        "abstract": normalize_text(abstract),
        "full_text": normalized,
        "sentences": sentences,
        "sections": sections,
    }
