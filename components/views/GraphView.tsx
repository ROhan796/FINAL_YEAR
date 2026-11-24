'use client';

import React, { useEffect, useMemo, useState } from 'react';
import {
  GitBranch,
  Sparkles,
  ThumbsUp,
  ThumbsDown,
  RefreshCw,
  Database,
  Compass,
  DownloadCloud,
  Layers,
} from 'lucide-react';
import { graphApi, insightsApi } from '@/lib/api';
import type { Discovery, GraphExploration, GraphNode } from '@/lib/api/graph';
import type { IngestStatus, ResearchGap } from '@/lib/api/insights';

const TYPE_COLORS: Record<string, string> = {
  METHOD: '#d97757',
  ALGORITHM: '#2563eb',
  DISEASE: '#dc2626',
  MATERIAL: '#059669',
  CHEMICAL: '#7c3aed',
  DATASET: '#d97706',
  PHENOMENON: '#0891b2',
  METRIC: '#4b5563',
};

export const GraphView: React.FC = () => {
  const [tab, setTab] = useState<'graph' | 'discoveries' | 'gaps' | 'ingest'>('discoveries');
  const [graph, setGraph] = useState<GraphExploration | null>(null);
  const [discoveries, setDiscoveries] = useState<Discovery[]>([]);
  const [gaps, setGaps] = useState<ResearchGap[]>([]);
  const [ingestStatus, setIngestStatus] = useState<IngestStatus | null>(null);

  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [hops, setHops] = useState(2);
  const [relationFilter, setRelationFilter] = useState('');
  const [loading, setLoading] = useState(false);
  const [retraining, setRetraining] = useState(false);
  const [retrainMsg, setRetrainMsg] = useState('');

  // Ingest form
  const [ingestSource, setIngestSource] = useState<'arxiv' | 'pubmed' | 'semantic-scholar'>('arxiv');
  const [ingestQuery, setIngestQuery] = useState('graph neural networks');
  const [ingestMsg, setIngestMsg] = useState('');

  const loadAll = async () => {
    setLoading(true);
    try {
      const [g, d, gp, st] = await Promise.allSettled([
        graphApi.exploration(selectedNode?.id, hops, relationFilter || undefined),
        graphApi.getDiscoveries(30),
        insightsApi.researchGaps(15),
        insightsApi.ingestStatus(),
      ]);
      if (g.status === 'fulfilled') setGraph(g.value);
      if (d.status === 'fulfilled') setDiscoveries(d.value);
      if (gp.status === 'fulfilled') setGaps(gp.value);
      if (st.status === 'fulfilled') setIngestStatus(st.value);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hops, relationFilter]);

  const handleExpandNode = async (node: GraphNode) => {
    setSelectedNode(node);
    setLoading(true);
    try {
      const g = await graphApi.exploration(node.id, hops, relationFilter || undefined);
      setGraph(g);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (id: string, useful: boolean) => {
    try {
      const updated = await graphApi.feedback(id, useful);
      setDiscoveries((prev) =>
        useful ? prev.map((d) => (d.id === id ? updated : d)) : prev.filter((d) => d.id !== id),
      );
    } catch (err) {
      console.error('Feedback failed:', err);
    }
  };

  const handleRetrain = async () => {
    setRetraining(true);
    setRetrainMsg('');
    try {
      const res = await graphApi.retrain(true, 25);
      setRetrainMsg(
        `GNN trained (${res.discoveriesCount} discoveries, Hits@10=${res.hitsAt10.toFixed(2)}, MRR=${res.mrr.toFixed(2)})`,
      );
      const fresh = await graphApi.getDiscoveries(30);
      setDiscoveries(fresh);
    } catch (err) {
      setRetrainMsg(err instanceof Error ? err.message : 'Retrain failed');
    } finally {
      setRetraining(false);
    }
  };

  const handleIngest = async (e: React.FormEvent) => {
    e.preventDefault();
    setIngestMsg('Dispatching ingestion + SPECTER2 embedding + KG build...');
    try {
      const res = await insightsApi.triggerIngest(ingestSource, ingestQuery, 15);
      setIngestMsg(`Queued ${res.source} ingest (${res.backend}${res.jobId ? ` #${res.jobId}` : ''}). Refreshing status shortly.`);
      setTimeout(() => {
        insightsApi.ingestStatus().then(setIngestStatus).catch(() => {});
      }, 3000);
    } catch (err) {
      setIngestMsg(err instanceof Error ? err.message : 'Ingest failed');
    }
  };

  // Deterministic circular/force-inspired SVG layout
  const layout = useMemo(() => {
    if (!graph || !graph.nodes.length) return { positions: new Map<string, { x: number; y: number }>() };
    const W = 760;
    const H = 420;
    const cx = W / 2;
    const cy = H / 2;
    const nodes = graph.nodes.slice(0, 48);
    const positions = new Map<string, { x: number; y: number }>();

    nodes.forEach((n, idx) => {
      if (graph.centerId && n.id === graph.centerId) {
        positions.set(n.id, { x: cx, y: cy });
      } else {
        const ring = idx % 2 === 0 ? 150 : 95;
        const angle = (2 * Math.PI * idx) / Math.max(1, nodes.length);
        positions.set(n.id, {
          x: cx + ring * 1.45 * Math.cos(angle),
          y: cy + ring * Math.sin(angle),
        });
      }
    });
    return { positions };
  }, [graph]);

  return (
    <div className="space-y-6 max-w-6xl mx-auto pb-12">
      {/* Banner */}
      <div className="bg-[#f5e3c7] border border-[#cccbc8] rounded-[24px] p-6 md:p-8 flex items-center justify-between flex-wrap gap-4">
        <div className="space-y-2">
          <span className="font-sans text-xs font-bold uppercase tracking-wider text-[#d97757] flex items-center gap-1.5">
            <GitBranch className="w-3.5 h-3.5" /> CKGAI · GNN + Knowledge Graph
          </span>
          <h1 className="font-serif text-3xl font-normal text-[#141413]">
            Cross-Domain Knowledge Graph & Discovery
          </h1>
          <p className="font-serif text-sm text-[#141413]/85 max-w-2xl">
            Explore extracted scientific entities, PyTorch R-GCN + GAT link predictions with Platt-calibrated confidence, and cross-domain research gaps.
          </p>
        </div>

        <div className="flex items-center gap-2 font-sans text-xs">
          <button
            onClick={handleRetrain}
            disabled={retraining}
            className="btn-clay py-2.5 px-4 flex items-center gap-1.5"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${retraining ? 'animate-spin' : ''}`} />
            <span>{retraining ? 'Training GNN...' : 'Train GNN & Discover'}</span>
          </button>
        </div>
      </div>

      {retrainMsg && (
        <div className="bg-[#faf9f5] border border-[#d97757] rounded-xl px-4 py-2.5 font-sans text-xs text-[#141413]">
          {retrainMsg}
        </div>
      )}

      {/* Stats Strip */}
      {ingestStatus && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 font-sans">
          {[
            ['Papers Indexed', ingestStatus.totalPapers],
            ['SPECTER2 Vectors', ingestStatus.embeddedPapers],
            ['KG Entities', ingestStatus.kgNodes],
            ['KG Relations', ingestStatus.kgEdges],
            ['Discoveries', ingestStatus.discoveries],
          ].map(([label, val]) => (
            <div key={String(label)} className="bg-[#faf9f5] border border-[#cccbc8] rounded-2xl p-3.5">
              <div className="text-[10px] font-bold uppercase tracking-wider text-[#87867f]">{label}</div>
              <div className="font-serif text-2xl text-[#141413] mt-0.5">{val}</div>
            </div>
          ))}
        </div>
      )}

      {/* Sub-navigation */}
      <div className="flex border-b border-[#cccbc8] gap-2 overflow-x-auto">
        {[
          { id: 'discoveries', label: `Cross-Domain Discoveries (${discoveries.length})`, icon: Sparkles },
          { id: 'graph', label: `Interactive Graph (${graph?.nodes.length || 0} nodes)`, icon: GitBranch },
          { id: 'gaps', label: `Research Gaps (${gaps.length})`, icon: Compass },
          { id: 'ingest', label: 'Corpus Ingestion & Queue', icon: DownloadCloud },
        ].map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => setTab(item.id as typeof tab)}
              className={`font-sans text-xs px-4 py-2.5 font-medium border-b-2 flex items-center gap-1.5 whitespace-nowrap cursor-pointer ${
                tab === item.id
                  ? 'border-[#d97757] text-[#d97757] font-semibold'
                  : 'border-transparent text-[#87867f] hover:text-[#141413]'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>

      {/* 1. DISCOVERIES */}
      {tab === 'discoveries' && (
        <div className="space-y-4">
          {discoveries.length === 0 ? (
            <div className="bg-[#faf9f5] border border-[#cccbc8] rounded-[24px] p-10 text-center space-y-3">
              <Sparkles className="w-8 h-8 text-[#d97757] mx-auto" />
              <p className="font-serif text-base text-[#141413]">
                No cross-domain discoveries generated yet. Click &ldquo;Train GNN &amp; Discover&rdquo; above.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {discoveries.map((d) => (
                <div
                  key={d.id}
                  className="bg-[#faf9f5] border border-[#cccbc8] rounded-[24px] p-5 space-y-3 hover:border-[#141413] transition-all"
                >
                  <div className="flex items-center justify-between gap-2 font-sans text-xs">
                    <span className="px-2.5 py-0.5 rounded-full bg-[#f5e3c7] text-[#d97757] font-bold">
                      {d.crossDomain ? 'Cross-Domain Prediction' : 'Link Prediction'}
                    </span>
                    <span className="font-mono font-bold text-[#141413]">
                      Confidence: {(d.score * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="bg-[#f0eee6] border border-[#cccbc8] rounded-xl p-3 flex items-center justify-between gap-2 text-xs">
                    <div>
                      <div className="font-serif text-base font-medium text-[#141413]">{d.srcNode.label}</div>
                      <div className="font-sans text-[10px] text-[#87867f]">
                        {d.srcNode.entityType} • {d.srcNode.domain || 'General'}
                      </div>
                    </div>
                    <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-[#141413] text-[#faf9f5] shrink-0">
                      {d.relation} →
                    </span>
                    <div className="text-right">
                      <div className="font-serif text-base font-medium text-[#141413]">{d.dstNode.label}</div>
                      <div className="font-sans text-[10px] text-[#87867f]">
                        {d.dstNode.entityType} • {d.dstNode.domain || 'General'}
                      </div>
                    </div>
                  </div>

                  {d.evidencePaths.length > 0 && (
                    <div className="font-sans text-[11px] text-[#87867f]">
                      Path: <span className="text-[#141413]">{d.evidencePaths[0].join(' → ')}</span>
                    </div>
                  )}

                  <div className="flex items-center justify-between pt-2 border-t border-[#cccbc8]/60 font-sans text-xs">
                    <span className="text-[11px] text-[#87867f]">
                      {d.evidencePaperIds.length} supporting paper(s)
                    </span>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleFeedback(d.id, true)}
                        className="px-2.5 py-1 rounded-lg border border-[#cccbc8] hover:bg-emerald-50 hover:border-emerald-400 flex items-center gap-1 cursor-pointer"
                      >
                        <ThumbsUp className="w-3.5 h-3.5 text-emerald-700" />
                        <span>Useful</span>
                      </button>
                      <button
                        onClick={() => handleFeedback(d.id, false)}
                        className="px-2.5 py-1 rounded-lg border border-[#cccbc8] hover:bg-rose-50 hover:border-rose-300 flex items-center gap-1 cursor-pointer"
                      >
                        <ThumbsDown className="w-3.5 h-3.5 text-rose-700" />
                        <span>Dismiss</span>
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 2. INTERACTIVE GRAPH */}
      {tab === 'graph' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-8 bg-[#faf9f5] border border-[#cccbc8] rounded-[24px] p-4 space-y-3">
            <div className="flex items-center justify-between flex-wrap gap-2 font-sans text-xs px-2">
              <div className="flex items-center gap-2">
                <span className="font-semibold">Hops:</span>
                {[1, 2, 3].map((h) => (
                  <button
                    key={h}
                    onClick={() => setHops(h)}
                    className={`px-2.5 py-1 rounded-lg border ${
                      hops === h ? 'bg-[#141413] text-[#faf9f5]' : 'bg-[#f0eee6] border-[#cccbc8]'
                    }`}
                  >
                    {h}
                  </button>
                ))}
              </div>
              <select
                value={relationFilter}
                onChange={(e) => setRelationFilter(e.target.value)}
                className="bg-[#f0eee6] border border-[#cccbc8] rounded-lg px-2.5 py-1 text-xs"
              >
                <option value="">All Relations</option>
                {['used_in', 'derived_from', 'applied_to', 'improves', 'treats', 'causes', 'measures', 'contains'].map(
                  (r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ),
                )}
              </select>
              {selectedNode && (
                <button
                  onClick={() => {
                    setSelectedNode(null);
                    loadAll();
                  }}
                  className="text-[#d97757] underline"
                >
                  Reset Center
                </button>
              )}
            </div>

            <div className="bg-[#f0eee6] border border-[#cccbc8] rounded-2xl overflow-hidden">
              <svg viewBox="0 0 760 420" className="w-full h-[400px]">
                {graph?.edges.map((e) => {
                  const p1 = layout.positions.get(e.source);
                  const p2 = layout.positions.get(e.target);
                  if (!p1 || !p2) return null;
                  return (
                    <g key={e.id}>
                      <line
                        x1={p1.x}
                        y1={p1.y}
                        x2={p2.x}
                        y2={p2.y}
                        stroke="#b7b5ae"
                        strokeWidth={1.2}
                        strokeOpacity={0.75}
                      />
                    </g>
                  );
                })}
                {graph?.nodes.slice(0, 48).map((n) => {
                  const pos = layout.positions.get(n.id);
                  if (!pos) return null;
                  const fill = TYPE_COLORS[n.entityType] || '#141413';
                  const isCenter = selectedNode?.id === n.id;
                  return (
                    <g
                      key={n.id}
                      onClick={() => handleExpandNode(n)}
                      className="cursor-pointer"
                    >
                      <circle
                        cx={pos.x}
                        cy={pos.y}
                        r={isCenter ? 14 : Math.min(12, 6 + Math.sqrt(n.degree || 1))}
                        fill={fill}
                        stroke={isCenter ? '#141413' : '#faf9f5'}
                        strokeWidth={isCenter ? 3 : 1.5}
                      />
                      <text
                        x={pos.x}
                        y={pos.y + 18}
                        textAnchor="middle"
                        fontSize="10"
                        fill="#141413"
                        className="font-sans select-none"
                      >
                        {n.label.slice(0, 18)}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          </div>

          <div className="lg:col-span-4 bg-[#faf9f5] border border-[#cccbc8] rounded-[24px] p-5 space-y-4">
            <h3 className="font-sans text-xs font-bold uppercase tracking-wider text-[#141413] flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-[#d97757]" /> Node Inspector
            </h3>
            {selectedNode ? (
              <div className="space-y-2 text-xs font-sans">
                <div className="font-serif text-xl text-[#141413]">{selectedNode.label}</div>
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded bg-[#f5e3c7] text-[#141413] font-semibold">
                    {selectedNode.entityType}
                  </span>
                  <span className="text-[#87867f]">{selectedNode.domain || 'General'}</span>
                </div>
                <div className="text-[#87867f]">Degree / Mentions: {selectedNode.degree}</div>
              </div>
            ) : (
              <p className="font-serif text-xs text-[#87867f]">
                Click any node in the graph to inspect its type, domain, and expand its {hops}-hop neighborhood.
              </p>
            )}
            <div className="pt-3 border-t border-[#cccbc8] space-y-1.5">
              <div className="font-sans text-[11px] font-bold text-[#87867f] uppercase">Entity Legend</div>
              <div className="flex flex-wrap gap-1.5 font-sans text-[10px]">
                {Object.entries(TYPE_COLORS).map(([t, c]) => (
                  <span key={t} className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-[#f0eee6]">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: c }} />
                    {t}
                  </span>
                ))}
              </div>
            </div>
            {loading && <div className="font-sans text-xs text-[#d97757]">Loading subgraph...</div>}
          </div>
        </div>
      )}

      {/* 3. RESEARCH GAPS */}
      {tab === 'gaps' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {gaps.map((g) => (
            <div key={g.id} className="bg-[#faf9f5] border border-[#cccbc8] rounded-[24px] p-5 space-y-2">
              <div className="flex items-center justify-between font-sans text-xs">
                <span className="px-2.5 py-0.5 rounded-full bg-[#f5e3c7] text-[#d97757] font-bold">
                  Unexplored Cross-Domain Gap
                </span>
                <span className="font-mono font-bold">Score {(g.gapScore * 100).toFixed(0)}%</span>
              </div>
              <h3 className="font-serif text-lg text-[#141413]">
                {g.method} ({g.methodDomain}) × {g.target} ({g.targetDomain})
              </h3>
              <p className="font-serif text-xs text-[#87867f] leading-relaxed">{g.summary}</p>
            </div>
          ))}
        </div>
      )}

      {/* 4. INGESTION & QUEUE */}
      {tab === 'ingest' && (
        <div className="bg-[#faf9f5] border border-[#cccbc8] rounded-[24px] p-6 space-y-4">
          <div className="flex items-center gap-2">
            <Database className="w-5 h-5 text-[#d97757]" />
            <h3 className="font-serif text-xl text-[#141413]">Live Corpus Ingestion (arXiv · PubMed · Semantic Scholar)</h3>
          </div>
          <p className="font-serif text-xs text-[#87867f]">
            Fetch real papers, chunk abstracts for RAG, compute SPECTER2 embeddings, and extract entities/relations into the Knowledge Graph.
          </p>
          <form onSubmit={handleIngest} className="flex flex-wrap items-center gap-3 font-sans text-xs">
            <select
              value={ingestSource}
              onChange={(e) => setIngestSource(e.target.value as typeof ingestSource)}
              className="p-2.5 bg-[#f0eee6] border border-[#cccbc8] rounded-xl"
            >
              <option value="arxiv">arXiv</option>
              <option value="pubmed">PubMed (Biomedical)</option>
              <option value="semantic-scholar">Semantic Scholar</option>
            </select>
            <input
              type="text"
              required
              value={ingestQuery}
              onChange={(e) => setIngestQuery(e.target.value)}
              placeholder="Search query to ingest..."
              className="flex-1 min-w-[220px] p-2.5 bg-[#f0eee6] border border-[#cccbc8] rounded-xl"
            />
            <button type="submit" className="btn-clay py-2.5 px-5">
              Ingest + Embed + Index KG
            </button>
          </form>
          {ingestMsg && <div className="font-sans text-xs text-[#d97757] font-medium">{ingestMsg}</div>}
        </div>
      )}
    </div>
  );
};
