#!/usr/bin/env python3
"""Seed development data into the database.

Uses the 5 papers and 5 researchers from the frontend mock data.
Generates PLACEHOLDER embeddings (random vectors, not real model output).

Usage:
    cd backend
    python -m scripts.seed
"""

from __future__ import annotations

import sys
import os
import uuid
import random
from datetime import UTC, datetime

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import get_engine
from app.db.base import Base, new_uuid, utcnow
import app.models  # register all tables


def seed():
    """Seed the database with development data."""
    engine = get_engine()
    Base.metadata.create_all(engine)

    from sqlalchemy.orm import Session
    session = Session(engine)

    try:
        # ── Users ──────────────────────────────────────────────────────
        from app.models.user import User, UserProfile

        # Dev auth middleware uses id="current-user" — ensure it exists.
        dev_user = User(
            id="current-user",
            email="dev@ckgai.local",
            password_hash="$2b$12$dev.placeholder",
            full_name="Dev User",
            role="verified_researcher",
            verified=True,
        )
        session.merge(dev_user)
        session.merge(UserProfile(
            user_id="current-user",
            title="Developer",
            institution="CKGAI Lab",
            field="Computer Science",
            bio="Development user.",
        ))

        user_id = "user-001"
        user = User(
            id=user_id,
            email="demo@ckgai.dev",
            password_hash="$2b$12$demo.hash.placeholder",
            full_name="Dr. Amara Nwosu",
            role="verified_researcher",
            verified=True,
        )
        session.merge(user)

        profile = UserProfile(
            user_id=user_id,
            title="Senior Researcher",
            institution="MIT CSAIL",
            field="Artificial Intelligence",
            bio="Researching the intersection of machine learning and scientific discovery.",
            avatar="https://picsum.photos/seed/user001/200",
            interests=["machine learning", "knowledge graphs", "scientific discovery"],
            publications_count=42,
            citation_count=1850,
            h_index=18,
        )
        session.merge(profile)
        session.commit()
        print(f"Seeded user: {user_id}")

        # ── Papers ─────────────────────────────────────────────────────
        papers_data = [
            {
                "id": "paper-101",
                "title": "Attention Is All You Need",
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
                "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
                "year": 2017,
                "venue": "NeurIPS",
                "doi": "10.48550/arXiv.1706.03762",
                "tags": ["deep learning", "transformers", "attention mechanism", "NLP"],
                "citation_count": 120000,
                "open_access": True,
                "domain": "Computer Science",
            },
            {
                "id": "paper-102",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
                "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
                "year": 2019,
                "venue": "NAACL",
                "doi": "10.18653/v1/N19-1423",
                "tags": ["NLP", "BERT", "pre-training", "language models"],
                "citation_count": 85000,
                "open_access": True,
                "domain": "Computer Science",
            },
            {
                "id": "paper-103",
                "title": "Graph Attention Networks",
                "abstract": "We present graph attention networks (GATs), novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their generalizations.",
                "authors": ["Petar Veličković", "Guillem Cucurull", "Adina Casanova"],
                "year": 2018,
                "venue": "ICLR",
                "doi": "10.48550/arXiv.1710.10903",
                "tags": ["graph neural networks", "attention", "GNN", "deep learning"],
                "citation_count": 25000,
                "open_access": True,
                "domain": "Computer Science",
            },
            {
                "id": "paper-104",
                "title": "CRISPR-Cas9 Gene Editing for Sickle Cell Disease and Beta-Thalassemia",
                "abstract": "We report the safety and efficacy of therapy with autologous CRISPR-Cas9-modified CD34+ cells in patients with sickle cell disease or transfusion-dependent beta-thalassemia. Editing of the BCL11A erythroid enhancer produced durable engraftment of edited cells with sustained fetal hemoglobin production.",
                "authors": ["Frangoul et al."],
                "year": 2021,
                "venue": "NEJM",
                "doi": "10.1056/NEJMoa2031054",
                "tags": ["CRISPR", "gene editing", "sickle cell", "hematological diseases"],
                "citation_count": 3500,
                "open_access": False,
                "domain": "Medicine",
            },
            {
                "id": "paper-105",
                "title": "Graphene-Based Biosensors for Early Disease Detection",
                "abstract": "We present a novel graphene-based biosensor platform for early detection of biomarkers associated with neurodegenerative diseases. The sensor demonstrates picomolar sensitivity for beta-amyloid detection using functionalized graphene field-effect transistors.",
                "authors": ["S. Park", "J. Kim", "M. Lee"],
                "year": 2023,
                "venue": "ACS Nano",
                "doi": "10.1021/acsnano.3c01234",
                "tags": ["graphene", "biosensor", "neurodegenerative diseases", "nanotechnology"],
                "citation_count": 150,
                "open_access": True,
                "domain": "Materials Science",
            },
        ]

        from app.models.paper import Paper, PaperTag
        for p in papers_data:
            paper = Paper(
                id=p["id"],
                source="seed",
                title=p["title"],
                abstract=p["abstract"],
                authors=p["authors"],
                year=p["year"],
                venue=p["venue"],
                doi=p["doi"],
                tags=p["tags"],
                citation_count=p["citation_count"],
                open_access=p["open_access"],
                domain=p["domain"],
                # PLACEHOLDER embeddings — NOT real model output
                embedding=[random.random() for _ in range(768)],
            )
            session.merge(paper)
            for tag in p["tags"]:
                session.merge(PaperTag(paper_id=p["id"], tag=tag))
        session.commit()
        print(f"Seeded {len(papers_data)} papers")

        # ── Researchers ────────────────────────────────────────────────
        from app.models.social import ResearcherProfile
        researchers_data = [
            {
                "id": "researcher-001",
                "name": "Dr. Amara Nwosu",
                "title": "Senior Researcher",
                "institution": "MIT CSAIL",
                "field": "Artificial Intelligence",
                "bio": "Specializing in knowledge graphs and scientific discovery.",
                "verified": True,
                "publications_count": 42,
                "citation_count": 1850,
                "h_index": 18,
                "interests": ["knowledge graphs", "NLP", "scientific discovery"],
            },
            {
                "id": "researcher-002",
                "name": "Dr. James Chen",
                "title": "Assistant Professor",
                "institution": "Stanford University",
                "field": "Computational Biology",
                "bio": "Applying machine learning to biological systems.",
                "verified": True,
                "publications_count": 28,
                "citation_count": 950,
                "h_index": 14,
                "interests": ["computational biology", "genomics", "ML"],
            },
            {
                "id": "researcher-003",
                "name": "Dr. Priya Sharma",
                "title": "Postdoctoral Fellow",
                "institution": "Max Planck Institute",
                "field": "Materials Science",
                "bio": "Researching nanomaterials for biomedical applications.",
                "verified": True,
                "publications_count": 15,
                "citation_count": 620,
                "h_index": 11,
                "interests": ["nanotechnology", "biosensors", "graphene"],
            },
            {
                "id": "researcher-004",
                "name": "Dr. Alex Petrov",
                "title": "Research Scientist",
                "institution": "DeepMind",
                "field": "Machine Learning",
                "bio": "Working on graph neural networks for scientific reasoning.",
                "verified": True,
                "publications_count": 35,
                "citation_count": 2200,
                "h_index": 20,
                "interests": ["GNN", "reinforcement learning", "scientific reasoning"],
            },
            {
                "id": "researcher-005",
                "name": "Dr. Sarah Kim",
                "title": "Associate Professor",
                "institution": "Harvard Medical School",
                "field": "Neuroscience",
                "bio": "Studying neurodegenerative disease mechanisms and diagnostics.",
                "verified": True,
                "publications_count": 50,
                "citation_count": 3100,
                "h_index": 24,
                "interests": ["neuroscience", "Alzheimer's", "biomarkers"],
            },
        ]

        for r in researchers_data:
            researcher = ResearcherProfile(
                id=r["id"],
                name=r["name"],
                title=r["title"],
                institution=r["institution"],
                field=r["field"],
                bio=r["bio"],
                avatar=f"https://picsum.photos/seed/{r['id']}/200",
                verified=r["verified"],
                publications_count=r["publications_count"],
                citation_count=r["citation_count"],
                h_index=r["h_index"],
                follower_count=random.randint(50, 500),
                interests=r["interests"],
            )
            session.merge(researcher)
        session.commit()
        print(f"Seeded {len(researchers_data)} researchers")

        # ── KG Nodes (from paper entities) ─────────────────────────────
        from app.models.knowledge import KgNode, KgEdge
        kg_entities = [
            ("Transformer", "METHOD", "Computer Science"),
            ("BERT", "METHOD", "Computer Science"),
            ("CRISPR-Cas9", "METHOD", "Medicine"),
            ("Graphene", "MATERIAL", "Materials Science"),
            ("Attention Mechanism", "ALGORITHM", "Computer Science"),
            ("Graph Neural Network", "METHOD", "Computer Science"),
            ("Biosensor", "METHOD", "Materials Science"),
            ("Sickle Cell Disease", "DISEASE", "Medicine"),
            ("Beta-Amyloid", "CHEMICAL", "Medicine"),
            ("Neurodegenerative Disease", "DISEASE", "Medicine"),
            ("Fetal Hemoglobin", "CHEMICAL", "Medicine"),
            ("Field-Effect Transistor", "METHOD", "Materials Science"),
            ("BCL11A", "CHEMICAL", "Medicine"),
            ("Language Model", "ALGORITHM", "Computer Science"),
            ("Pre-training", "METHOD", "Computer Science"),
        ]

        node_ids = {}
        for name, etype, domain in kg_entities:
            nid = new_uuid()
            node_ids[name] = nid
            node = KgNode(
                id=nid,
                canonical_name=name,
                normalised_name=name.lower(),
                entity_type=etype,
                domain=domain,
                embedding=[random.random() for _ in range(768)],
                confidence=0.9,
                mention_count=random.randint(1, 50),
            )
            session.merge(node)
        session.commit()
        print(f"Seeded {len(kg_entities)} KG nodes")

        # ── KG Edges ──────────────────────────────────────────────────
        kg_edges = [
            ("Transformer", "used_in", "Language Model"),
            ("BERT", "derived_from", "Transformer"),
            ("BERT", "uses", "Attention Mechanism"),
            ("CRISPR-Cas9", "treats", "Sickle Cell Disease"),
            ("CRISPR-Cas9", "improves", "Fetal Hemoglobin"),
            ("Graphene", "used_in", "Biosensor"),
            ("Biosensor", "measures", "Beta-Amyloid"),
            ("Beta-Amyloid", "causes", "Neurodegenerative Disease"),
            ("Graph Neural Network", "used_in", "Transformer"),
            ("BCL11A", "measures", "Fetal Hemoglobin"),
            ("Field-Effect Transistor", "contains", "Graphene"),
        ]

        for src_name, rel, dst_name in kg_edges:
            if src_name in node_ids and dst_name in node_ids:
                edge = KgEdge(
                    id=new_uuid(),
                    src_id=node_ids[src_name],
                    dst_id=node_ids[dst_name],
                    relation=rel,
                    confidence=0.85,
                    src_domain="",
                    dst_domain="",
                    domain_cross=False,
                )
                session.merge(edge)
        session.commit()
        print(f"Seeded {len(kg_edges)} KG edges")

        print("\n✓ Development seed data loaded successfully")
        print("  NOTE: Embeddings are PLACEHOLDER random vectors")
        print("  Run 'python -m ml.pipelines.index' with real models for actual embeddings")

    except Exception as e:
        session.rollback()
        print(f"Error seeding data: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    seed()
