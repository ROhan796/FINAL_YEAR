'use client';

import React from 'react';
import Link from 'next/link';
import {
  Sparkles,
  ArrowRight,
  Search,
  Network,
  FolderKanban,
  Bookmark,
  ShieldCheck,
  GitBranch,
  Layers,
} from 'lucide-react';

export const LandingView: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#f0eee6] text-[#141413] font-serif">
      {/* Public Top Bar */}
      <header className="border-b border-[#cccbc8] bg-[#faf9f5]/90 backdrop-blur-xs sticky top-0 z-30">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-[#141413] text-[#faf9f5] flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-[#d97757]" />
            </div>
            <span className="font-sans text-base font-bold lowercase tracking-tight">
              suiiiiiiii <span className="text-[#87867f] font-normal">/ ckgai</span>
            </span>
          </div>

          <div className="flex items-center gap-3 font-sans text-xs">
            <Link href="/login" className="px-4 py-2 rounded-xl border border-[#cccbc8] bg-[#faf9f5] hover:border-[#141413] text-[#141413] font-medium no-underline">
              Sign In
            </Link>
            <Link href="/register" className="px-4 py-2 rounded-xl border border-[#141413] bg-[#faf9f5] hover:bg-[#e3dacc] text-[#141413] font-semibold no-underline">
              Get Started
            </Link>
            <Link href="/feed" className="btn-clay px-4 py-2 text-xs flex items-center gap-1.5 no-underline">
              <span>Launch Platform</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <main className="max-w-6xl mx-auto px-6 py-16 space-y-16">
        <section className="bg-[#faf9f5] border border-[#cccbc8] rounded-[28px] p-8 md:p-14 grid grid-cols-1 lg:grid-cols-12 gap-10 items-center">
          <div className="lg:col-span-7 space-y-6">
            <span className="inline-flex items-center gap-1.5 font-sans text-xs font-bold uppercase tracking-wider text-[#d97757] bg-[#f5e3c7] px-3 py-1 rounded-full">
              <GitBranch className="w-3.5 h-3.5" /> Cross-Domain Knowledge Graph AI
            </span>
            <h1 className="text-4xl md:text-5xl font-normal leading-tight text-[#141413]">
              Your library, collaborators, knowledge graph, and writing pipeline — in one brain.
            </h1>
            <p className="text-base text-[#141413]/80 leading-relaxed max-w-2xl">
              Researchers live across five disconnected tools. CKGAI unifies semantic paper retrieval (SPECTER2 + Matryoshka), entity-relation knowledge graphs (SciBERT + CRF), GNN cross-domain link prediction (R-GCN + GAT), and a RAG-grounded workspace copilot.
            </p>
            <div className="flex flex-wrap items-center gap-3 pt-2 font-sans">
              <Link href="/feed" className="btn-clay py-3 px-6 text-xs flex items-center gap-2 no-underline">
                <span>Enter Research Feed</span>
                <ArrowRight className="w-4 h-4" />
              </Link>
              <Link href="/graph" className="btn-outlined-dark py-3 px-6 text-xs flex items-center gap-2 no-underline">
                <Network className="w-4 h-4 text-[#d97757]" />
                <span>Explore Knowledge Graph</span>
              </Link>
              <Link href="/search" className="btn-outlined-dark py-3 px-6 text-xs flex items-center gap-2 no-underline">
                <Search className="w-4 h-4 text-[#d97757]" />
                <span>Try Semantic Search</span>
              </Link>
            </div>
          </div>

          <div className="lg:col-span-5 bg-[#f5e3c7] border border-[#cccbc8] rounded-2xl p-6 space-y-4">
            <span className="font-sans text-[11px] font-bold uppercase tracking-wider text-[#d97757]">
              Continuous Research Workflow
            </span>
            <div className="space-y-2.5 font-sans text-xs">
              {[
                ['1. Ingest & Embed', 'arXiv, PubMed & Semantic Scholar -> SPECTER2 768-d / 128-d MRL'],
                ['2. Understand & Resolve', 'SciBERT+CRF NER -> Relation Extraction -> Provenance KG'],
                ['3. Discover Across Fields', 'PyTorch R-GCN + GAT + DistMult -> Platt-calibrated links'],
                ['4. Collaborate & Publish', 'RAG SSE Copilot, Reading Room Annotations & Citation Gap Check'],
              ].map(([title, desc]) => (
                <div key={title} className="bg-[#faf9f5] border border-[#cccbc8] rounded-xl p-3">
                  <div className="font-bold text-[#141413]">{title}</div>
                  <div className="text-[#87867f] text-[11px] mt-0.5">{desc}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Core Capabilities Grid */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            {
              icon: Search,
              title: 'Hybrid Semantic Search + Explainability',
              desc: 'Vector (SPECTER2 + Matryoshka), full-text, and KG arms fused with RRF and temporal trend boosting — with a "Why This Paper?" provenance sidebar.',
            },
            {
              icon: Network,
              title: 'GNN Cross-Domain Discovery',
              desc: 'Trained PyTorch R-GCN + GAT + Virtual Node predicts high-value cross-domain links and research gaps with interactive force-graph exploration.',
            },
            {
              icon: FolderKanban,
              title: 'RAG Copilot & Reading Room',
              desc: 'Workspace assistant streams answers grounded in pinned paper chunks via SSE, alongside shared sentence-level annotations and Research Velocity.',
            },
            {
              icon: Bookmark,
              title: 'Smart Collections & Debt Tracker',
              desc: 'K-Means + TF-IDF automatic topic clustering, Cross-Collection bridging papers, and a Research Debt tracker for aging unread saves.',
            },
            {
              icon: ShieldCheck,
              title: 'Manuscript Pre-Review & Citation Gaps',
              desc: 'AI pre-review scoring paired with NER-based Citation Gap detection that surfaces un-cited methods with candidate references.',
            },
            {
              icon: Layers,
              title: 'Research-Native Social Network',
              desc: 'Collaboration Affinity scoring based on KG entity overlap and shared topics — built around scientific threads, not vanity metrics.',
            },
          ].map((cap) => {
            const Icon = cap.icon;
            return (
              <div key={cap.title} className="bg-[#faf9f5] border border-[#cccbc8] rounded-[24px] p-6 space-y-3">
                <div className="w-10 h-10 rounded-xl bg-[#f5e3c7] flex items-center justify-center text-[#d97757]">
                  <Icon className="w-5 h-5" />
                </div>
                <h3 className="text-lg font-medium text-[#141413]">{cap.title}</h3>
                <p className="text-xs font-sans text-[#87867f] leading-relaxed">{cap.desc}</p>
              </div>
            );
          })}
        </section>
      </main>
    </div>
  );
};
