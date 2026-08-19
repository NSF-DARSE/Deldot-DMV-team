import React, { useEffect, useState } from "react";
import { useParams, Link, useNavigate, useLocation } from "react-router-dom";
import {
  ArrowLeft, Flag, ShieldCheck, HelpCircle, MessageSquare, Info,
  MapPin, IdCard, Car, Briefcase, Radio, ChevronRight, TrendingUp, TrendingDown, Minus, Tag as TagIcon, X, Plus, GitCompare
} from "lucide-react";
import { fetchCase, updateCase, addTag, removeTag, CLASSIFICATIONS, REVIEWER_STATUS_META, priorityBand, SUGGESTED_TAGS } from "../lib/api";
import PriorityGauge from "../components/PriorityGauge";
import ProbabilityBars from "../components/ProbabilityBars";
import FeatureImportance from "../components/FeatureImportance";
import { Button } from "../components/ui/button";
import { Textarea } from "../components/ui/textarea";
import { Input } from "../components/ui/input";
import { toast } from "sonner";

const EV_TYPES = [
  { key: "address", label: "Address Records", icon: MapPin, color: "#3B82F6" },
  { key: "credential", label: "Credential Records", icon: IdCard, color: "#8B5CF6" },
  { key: "vehicle_title", label: "Vehicle Title Records", icon: Car, color: "#06B6D4" },
  { key: "work", label: "Work Records", icon: Briefcase, color: "#F59E0B" },
  { key: "external", label: "External Records", icon: Radio, color: "#EC4899" },
];

const ACTIONS = [
  { key: "marked_for_review", label: "Mark for Review", icon: Flag, color: "bg-orange-500/20 hover:bg-orange-500/30 text-orange-300 border-orange-500/40" },
  { key: "cleared", label: "Clear Case", icon: ShieldCheck, color: "bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border-emerald-500/40" },
  { key: "info_requested", label: "Request More Information", icon: HelpCircle, color: "bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border-amber-500/40" },
];

function EvidenceCard({ type, records }) {
  const Icon = type.icon;
  const strong = records.filter(r => r.match_confidence >= 0.95).length;
  return (
    <div className="card-surface p-4 hover-elevate" data-testid={`evidence-${type.key}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-md flex items-center justify-center" style={{ background: `${type.color}18`, border: `1px solid ${type.color}44` }}>
            <Icon className="w-4 h-4" style={{ color: type.color }} />
          </div>
          <div>
            <div className="text-xs font-semibold text-white">{type.label}</div>
            <div className="text-[10px] uppercase tracking-widest text-slate-500">{records.length} record{records.length===1?"":"s"} · {strong} strong</div>
          </div>
        </div>
      </div>
      {records.length === 0 && <div className="text-xs text-slate-500 py-4 text-center">No records</div>}
      <div className="space-y-1.5 max-h-52 overflow-y-auto">
        {records.slice(0, 20).map((r, i) => (
          <div key={i} className="text-xs px-3 py-2 rounded-md bg-[#0F141C] border border-[#1B222E] flex items-center justify-between gap-2">
            <div className="min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="mono font-semibold text-slate-200">{r.state}</span>
                <span className="text-slate-500">·</span>
                <span className="text-slate-400">{r.event_type || r.source_domain}</span>
              </div>
              <div className="text-[11px] text-slate-500 mt-0.5 mono truncate">{r.date} · {r.source_record_id}</div>
            </div>
            <div className="text-right shrink-0">
              <div className={`text-[10px] font-semibold ${r.phase_available === "T1" ? "text-emerald-300" : "text-slate-500"}`}>{r.phase_available}</div>
              <div className="text-[10px] mono text-slate-500">c {r.match_confidence.toFixed(2)}</div>
            </div>
          </div>
        ))}
        {records.length > 20 && <div className="text-[10px] text-center text-slate-500 py-1">…and {records.length - 20} more</div>}
      </div>
    </div>
  );
}

function Delta({ prev, cur, format = (x) => x, higherIsWorse = true }) {
  if (typeof cur !== "number") {
    if (prev === cur) return <Minus className="w-3.5 h-3.5 text-slate-500 inline" />;
    return <ChevronRight className="w-3.5 h-3.5 text-amber-400 inline" />;
  }
  const d = cur - prev;
  if (Math.abs(d) < 0.005) return <span className="text-[11px] text-slate-500 mono">—</span>;
  const bad = higherIsWorse ? d > 0 : d < 0;
  const Icon = d > 0 ? TrendingUp : TrendingDown;
  return (
    <span className={`inline-flex items-center gap-1 text-[11px] mono font-semibold ${bad ? "text-red-400" : "text-emerald-400"}`}>
      <Icon className="w-3 h-3" />{d > 0 ? "+" : ""}{format(d)}
    </span>
  );
}

function TagEditor({ tags, onAdd, onRemove }) {
  const [input, setInput] = useState("");
  const submit = (t) => { const v = (t || input).trim(); if (v) { onAdd(v); setInput(""); } };
  return (
    <div data-testid="tag-editor">
      <div className="flex items-center gap-2 flex-wrap mb-3">
        {(tags || []).length === 0 && <span className="text-xs text-slate-500">No tags yet</span>}
        {(tags || []).map((t) => (
          <span key={t} className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md border border-cyan-500/30 bg-cyan-500/10 text-cyan-300 text-xs font-medium" data-testid={`tag-${t}`}>
            <TagIcon className="w-3 h-3" />{t}
            <button onClick={() => onRemove(t)} data-testid={`tag-remove-${t}`} className="text-cyan-400 hover:text-white transition-colors"><X className="w-3 h-3" /></button>
          </span>
        ))}
      </div>
      <div className="flex items-center gap-2 mb-3">
        <Input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Add custom tag…"
          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); submit(); } }}
          className="h-9 bg-[#0F141C] border-[#1E2633] text-slate-200 placeholder:text-slate-500 flex-1" data-testid="tag-input" />
        <Button size="sm" onClick={() => submit()} data-testid="tag-add" disabled={!input.trim()}
          className="h-9 bg-cyan-600/20 hover:bg-cyan-600/30 border border-cyan-500/40 text-cyan-300">
          <Plus className="w-3.5 h-3.5 mr-1" /> Add
        </Button>
      </div>
      <div>
        <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-2">Suggested</div>
        <div className="flex flex-wrap gap-1.5">
          {SUGGESTED_TAGS.filter(s => !(tags || []).includes(s)).map((t) => (
            <button key={t} onClick={() => submit(t)} data-testid={`suggested-${t}`}
              className="text-[11px] px-2 py-1 rounded border border-[#232C3B] bg-[#0F141C] text-slate-400 hover:text-cyan-300 hover:border-cyan-500/40 transition-colors">
              + {t}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function CaseDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const [c, setC] = useState(null);
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);

  const params = new URLSearchParams(location.search);
  const highlightKind = params.get("highlight") || "";
  const highlightQ = params.get("q") || "";

  const reload = async () => { const d = await fetchCase(id); setC(d); setNotes(d.reviewer_notes || ""); };
  useEffect(() => { reload(); }, [id]);
  useEffect(() => {
    if (!c || !highlightKind) return;
    const el = document.querySelector(`[data-testid="highlighted-evidence-${highlightKind}"]`);
    if (el) setTimeout(() => el.scrollIntoView({ behavior: "smooth", block: "center" }), 250);
  }, [c, highlightKind]);
  if (!c) return <div className="p-10 text-slate-500" data-testid="loading">Loading case…</div>;

  const cls = CLASSIFICATIONS[c.predicted_class];
  const prevCls = CLASSIFICATIONS[c.previous.predicted_class];
  const band = priorityBand(c.review_priority);
  const statusMeta = REVIEWER_STATUS_META[c.reviewer_status] || REVIEWER_STATUS_META.unreviewed;

  const apply = async (statusKey) => {
    setSaving(true);
    try { const u = await updateCase(id, { reviewer_status: statusKey, reviewer_notes: notes }); setC(u); toast.success("Action logged", { description: REVIEWER_STATUS_META[statusKey].label }); }
    catch { toast.error("Failed to save"); } finally { setSaving(false); }
  };
  const saveNotes = async () => {
    setSaving(true);
    try { const u = await updateCase(id, { reviewer_status: c.reviewer_status, reviewer_notes: notes }); setC(u); toast.success("Notes saved"); }
    catch { toast.error("Failed"); } finally { setSaving(false); }
  };
  const doAddTag = async (t) => {
    try { const s = await addTag(id, t); setC({ ...c, reviewer_tags: s.reviewer_tags }); toast.success(`Tag added: ${t}`); }
    catch { toast.error("Failed to add tag"); }
  };
  const doRemoveTag = async (t) => {
    try { const s = await removeTag(id, t); setC({ ...c, reviewer_tags: s.reviewer_tags }); toast.success(`Tag removed`); }
    catch { toast.error("Failed to remove tag"); }
  };

  const allRecords = Object.entries(c.evidence).flatMap(([kind, recs]) => recs.map(r => ({ ...r, kind })));
  const sorted = [...allRecords].sort((a, b) => (a.date < b.date ? -1 : 1));
  const timeline = [{ event: "T0 Initial Prediction", type: "phase", date: sorted[0]?.date || "—" }];
  sorted.slice(0, 20).forEach(r => timeline.push({ event: `${EV_TYPES.find(t => t.key === r.kind)?.label || r.kind} · ${r.state}`, type: "record", date: r.date, phase: r.phase_available }));
  if (sorted.length > 20) timeline.push({ event: `…and ${sorted.length - 20} more records`, type: "note", date: "" });
  timeline.push({ event: "T1 Updated Prediction", type: "phase", date: c.last_updated?.slice(0,10) || "—" });

  return (
    <div className="space-y-5" data-testid="case-detail-page">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={() => navigate(-1)} data-testid="back-button" className="text-slate-400 hover:text-white">
            <ArrowLeft className="w-4 h-4 mr-1" /> Back
          </Button>
          <div className="text-[10px] uppercase tracking-widest text-slate-500 font-medium">
            <Link to="/queue" className="hover:text-slate-200">Case Queue</Link> · Investigation
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Link to={`/compare?a=${c.candidate_id}`} data-testid="open-compare"
            className="inline-flex items-center gap-1.5 px-3 h-8 rounded-md bg-cyan-500/20 border border-cyan-500/40 text-cyan-300 text-xs font-semibold hover:bg-cyan-500/30 transition-colors">
            <GitCompare className="w-3.5 h-3.5" /> Compare
          </Link>
          <div className="text-xs mono text-slate-500">
            Last updated <span className="text-slate-300 font-semibold">{c.last_updated ? new Date(c.last_updated).toLocaleString() : "—"}</span>
          </div>
        </div>
      </div>

      <div className="card-surface-elevated p-5">
        <div className="flex items-start justify-between gap-6 flex-wrap">
          <div>
            <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Candidate Record ID</div>
            <h1 className="font-display text-3xl font-bold mono text-white" data-testid="case-candidate-id">{c.candidate_id}</h1>
            <div className="flex items-center gap-2 mt-2 flex-wrap">
              <span className={`inline-flex items-center px-2.5 py-1 rounded-md border text-xs font-semibold ${cls.bg} ${cls.text} ${cls.border}`} data-testid="case-class-badge">
                <span className="badge-dot" style={{ background: cls.color }} />{cls.label}
              </span>
              <span className="inline-flex items-center px-2.5 py-1 rounded-md border text-xs font-semibold border-blue-500/30 bg-blue-500/10 text-blue-300">Phase T1</span>
              <span className={`inline-flex items-center px-2.5 py-1 rounded-md border text-xs font-semibold ${band.bg} ${band.text} ${band.border}`}>{band.label} Priority</span>
              <span className="inline-flex items-center px-2.5 py-1 rounded-md border text-xs font-semibold border-[#232C3B] bg-[#121821] text-slate-300" data-testid="case-status-badge">
                <span className="badge-dot" style={{ background: statusMeta.color }} />{statusMeta.label}
              </span>
              <span className="inline-flex items-center px-2.5 py-1 rounded-md border text-xs font-semibold border-[#232C3B] bg-[#121821] text-slate-300 mono">
                Primary State: <span className="ml-1 text-white">{c.primary_state}</span>
              </span>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-6 text-right">
            <div><div className="text-[10px] uppercase tracking-widest text-slate-500">Confidence</div><div className="stat-value text-2xl text-white">{(c.confidence*100).toFixed(1)}%</div></div>
            <div><div className="text-[10px] uppercase tracking-widest text-slate-500">Priority</div><div className="stat-value text-2xl" style={{ color: band.color }}>{c.review_priority.toFixed(2)}</div></div>
            <div><div className="text-[10px] uppercase tracking-widest text-slate-500">New Evidence</div><div className="stat-value text-2xl text-blue-300">{c.new_evidence_count}</div></div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-5">
        <div className="col-span-12 lg:col-span-4 space-y-5">
          <div className="card-surface-elevated p-5">
            <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Priority</div>
            <h3 className="font-display text-base font-semibold text-white mb-2">Review Priority Index</h3>
            <PriorityGauge value={c.review_priority} />
          </div>
          <div className="card-surface-elevated p-5">
            <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Model Output</div>
            <h3 className="font-display text-base font-semibold text-white mb-4">Classification Probabilities</h3>
            <ProbabilityBars probs={c} predicted={c.predicted_class} />
          </div>
          <div className="card-surface-elevated p-5" data-testid="explanations">
            <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Explanation</div>
            <h3 className="font-display text-base font-semibold text-white mb-3">Why this case was classified this way</h3>
            <ul className="space-y-2">
              {c.explanations.map((ex, i) => (
                <li key={i} className="flex items-start gap-2 text-xs text-slate-300">
                  <div className="w-1.5 h-1.5 rounded-full mt-1.5 shrink-0 bg-blue-400" />
                  <span>{ex}</span>
                </li>
              ))}
            </ul>
            <div className="mt-4 p-3 rounded-md bg-amber-950/30 border border-amber-800/30 flex items-start gap-2 text-[11px] text-amber-200/80">
              <Info className="w-3.5 h-3.5 mt-0.5 shrink-0 text-amber-400" />
              <span><span className="font-semibold uppercase tracking-widest text-amber-300">Decision Support Only —</span> Not a determination of residency, violations, penalties, fees, guilt, or enforcement actions.</span>
            </div>
          </div>
          {/* Feature Importance */}
          <FeatureImportance candidateId={c.candidate_id} />
          {/* Tags */}
          <div className="card-surface-elevated p-5">
            <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Annotations</div>
            <h3 className="font-display text-base font-semibold text-white mb-4">Reviewer Tags</h3>
            <TagEditor tags={c.reviewer_tags} onAdd={doAddTag} onRemove={doRemoveTag} />
          </div>
        </div>

        <div className="col-span-12 lg:col-span-8 space-y-5">
          <div className="card-surface-elevated p-5" data-testid="t0-t1-comparison">
            <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Phase Comparison</div>
            <h3 className="font-display text-base font-semibold text-white mb-4">T0 → T1 Snapshot</h3>
            <div className="rounded-lg border border-[#1E2633] overflow-hidden">
              <div className="grid grid-cols-12 bg-[#0F141C] border-b border-[#1E2633] px-4 py-2 text-[10px] uppercase tracking-widest font-semibold text-slate-500">
                <div className="col-span-4">Attribute</div><div className="col-span-3">Previous (T0)</div>
                <div className="col-span-3">Current (T1)</div><div className="col-span-2 text-right">Change</div>
              </div>
              {[
                { k: "Recommendation", prev: prevCls?.short || c.previous.predicted_class, cur: cls?.short || c.predicted_class, prevColor: prevCls?.color, curColor: cls?.color },
                { k: "Review Priority", prev: c.previous.review_priority, cur: c.review_priority, fmt: (v) => v.toFixed(2), higherIsWorse: true, num: true },
                { k: "Confidence", prev: c.previous.confidence, cur: c.confidence, fmt: (v) => `${(v*100).toFixed(1)}%`, higherIsWorse: false, num: true },
                { k: "P(Review Warranted)", prev: c.previous.prob_review_warranted, cur: c.prob_review_warranted, fmt: (v) => `${(v*100).toFixed(0)}%`, higherIsWorse: true, num: true },
                { k: "P(Not Warranted)", prev: c.previous.prob_review_not_warranted, cur: c.prob_review_not_warranted, fmt: (v) => `${(v*100).toFixed(0)}%`, higherIsWorse: false, num: true },
                { k: "P(Insufficient)", prev: c.previous.prob_insufficient_information, cur: c.prob_insufficient_information, fmt: (v) => `${(v*100).toFixed(0)}%`, higherIsWorse: true, num: true },
              ].map((row, i) => (
                <div key={i} className="grid grid-cols-12 px-4 py-2.5 border-b border-[#151B25] last:border-b-0 items-center text-sm">
                  <div className="col-span-4 text-slate-400">{row.k}</div>
                  <div className="col-span-3 mono text-slate-400" style={row.prevColor ? { color: row.prevColor } : {}}>{row.num ? row.fmt(row.prev) : row.prev}</div>
                  <div className="col-span-3 mono font-semibold text-white" style={row.curColor ? { color: row.curColor } : {}}>{row.num ? row.fmt(row.cur) : row.cur}</div>
                  <div className="col-span-2 text-right">
                    {row.num ? <Delta prev={row.prev} cur={row.cur} format={row.fmt} higherIsWorse={row.higherIsWorse} />
                      : row.prev === row.cur ? <Minus className="w-3.5 h-3.5 text-slate-500 inline" /> : <ChevronRight className="w-3.5 h-3.5 text-amber-400 inline" />}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="card-surface-elevated p-5" data-testid="evidence-timeline">
            <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Chronology</div>
            <h3 className="font-display text-base font-semibold text-white mb-4">Evidence Timeline</h3>
            <div className="relative pl-6">
              <div className="absolute left-2 top-2 bottom-2 w-px bg-[#232C3B]"></div>
              {timeline.map((e, i) => (
                <div key={i} className="relative py-1.5">
                  <div className={`absolute -left-4 top-2 w-3 h-3 rounded-full border-2 ${
                    e.type === "phase" ? "bg-blue-500 border-blue-300" : e.phase === "T1" ? "bg-emerald-500 border-emerald-300" : e.type === "note" ? "bg-slate-700 border-slate-500" : "bg-slate-600 border-slate-400"
                  }`}></div>
                  <div className="flex items-center justify-between gap-3">
                    <div className={`text-xs ${e.type === "phase" ? "font-semibold text-white" : "text-slate-300"}`}>{e.event}</div>
                    <div className="text-[11px] mono text-slate-500">{e.date}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="card-surface-elevated p-5">
            <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Workflow</div>
            <h3 className="font-display text-base font-semibold text-white mb-1">Reviewer Actions & Notes</h3>
            <p className="text-xs text-slate-400 mb-4">Actions log to the audit trail. Notes are timestamped and preserved for review continuity.</p>
            <Textarea value={notes} onChange={(e) => setNotes(e.target.value)}
              placeholder="Add reviewer note: observed evidence, cross-checks performed, next actions to investigate…"
              className="min-h-[100px] mb-4 bg-[#0F141C] border-[#1E2633] text-slate-200" data-testid="reviewer-notes" />
            <div className="flex flex-wrap gap-2">
              {ACTIONS.map((a) => {
                const Icon = a.icon; const active = c.reviewer_status === a.key;
                return (
                  <button key={a.key} onClick={() => apply(a.key)} disabled={saving} data-testid={`action-${a.key}`}
                    className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-md border text-sm font-semibold transition-colors ${a.color} ${active ? "ring-2 ring-offset-2 ring-offset-[#161D27] ring-white/40" : ""} ${saving ? "opacity-60 cursor-not-allowed" : ""}`}>
                    <Icon className="w-4 h-4" /> {a.label}
                  </button>
                );
              })}
              <Button variant="outline" onClick={saveNotes} disabled={saving} data-testid="save-notes"
                className="ml-auto bg-[#0F141C] border-[#232C3B] text-slate-200 hover:bg-[#1B222E] hover:text-white">
                <MessageSquare className="w-4 h-4 mr-1.5" /> Add Reviewer Note
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div>
        <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-2">Evidence</div>
        <h3 className="font-display text-base font-semibold text-white mb-4">
          Evidence Records by Source
          {highlightKind && (
            <span className="ml-3 inline-flex items-center gap-1.5 px-2 py-0.5 rounded border border-cyan-500/40 bg-cyan-500/10 text-cyan-300 text-xs font-medium normal-case tracking-normal">
              Highlighted from search: {EV_TYPES.find(t => t.key === highlightKind)?.label || highlightKind}
              {highlightQ && <> · &ldquo;{highlightQ}&rdquo;</>}
            </span>
          )}
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {EV_TYPES.map((t) => (
            <div key={t.key} className={highlightKind === t.key ? "ring-2 ring-cyan-500/60 rounded-xl transition-shadow" : ""} data-testid={highlightKind === t.key ? `highlighted-evidence-${t.key}` : undefined}>
              <EvidenceCard type={t} records={c.evidence[t.key] || []} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
