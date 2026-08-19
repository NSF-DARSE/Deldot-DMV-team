import React, { useEffect, useState } from "react";
import { useSearchParams, Link } from "react-router-dom";
import { Search, GitCompare, ArrowRight, X, TrendingUp, TrendingDown, Minus, Info } from "lucide-react";
import { fetchCases, fetchCase, compareCases, CLASSIFICATIONS, priorityBand } from "../lib/api";
import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";

function CasePicker({ label, value, onSelect }) {
  const [q, setQ] = useState("");
  const [options, setOptions] = useState([]);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!q || q.length < 2) { setOptions([]); return; }
    let cancel = false;
    setLoading(true);
    fetchCases({ q, limit: 10, sort_by: "review_priority", sort_dir: "desc" }).then((d) => {
      if (!cancel) { setOptions(d.cases); setLoading(false); }
    });
    return () => { cancel = true; };
  }, [q]);
  return (
    <div className="card-surface p-4" data-testid={`picker-${label.toLowerCase().replace(" ", "-")}`}>
      <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-2">{label}</div>
      {value ? (
        <div className="flex items-center justify-between p-3 rounded-md bg-[#0F141C] border border-cyan-500/30">
          <div>
            <div className="mono font-semibold text-white">{value.candidate_id}</div>
            <div className="text-xs text-slate-400">{CLASSIFICATIONS[value.predicted_class]?.label} · Priority {value.review_priority.toFixed(2)}</div>
          </div>
          <button onClick={() => onSelect(null)} className="text-slate-400 hover:text-white"><X className="w-4 h-4" /></button>
        </div>
      ) : (
        <div>
          <div className="relative mb-2">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search candidate ID (CAN-…)"
              className="pl-9 h-10 bg-[#0F141C] border-[#1E2633] text-slate-200 placeholder:text-slate-500" data-testid={`picker-input-${label.toLowerCase().replace(" ", "-")}`} />
          </div>
          {loading && <div className="text-xs text-slate-500 py-2">Searching…</div>}
          <div className="space-y-1 max-h-64 overflow-y-auto">
            {options.map((o) => (
              <button key={o.candidate_id} onClick={() => onSelect(o)}
                className="w-full text-left p-2.5 rounded-md bg-[#0F141C] border border-[#1B222E] hover:border-cyan-500/40 transition-colors"
                data-testid={`option-${o.candidate_id}`}>
                <div className="flex items-center justify-between">
                  <span className="mono font-semibold text-white text-sm">{o.candidate_id}</span>
                  <span className="mono text-xs" style={{ color: priorityBand(o.review_priority).color }}>{o.review_priority.toFixed(2)}</span>
                </div>
                <div className="text-[11px] text-slate-500 mt-0.5">{CLASSIFICATIONS[o.predicted_class]?.label}</div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function DeltaCell({ prev, cur, format = (x) => x, higherIsWorse = true, num = true }) {
  if (!num) {
    if (prev === cur) return <Minus className="w-3.5 h-3.5 text-slate-500 inline" />;
    return <ChevronRight className="w-3.5 h-3.5 text-amber-400 inline" />;
  }
  const d = (cur ?? 0) - (prev ?? 0);
  if (Math.abs(d) < 0.005) return <span className="text-[11px] text-slate-500 mono">—</span>;
  const bad = higherIsWorse ? d > 0 : d < 0;
  const Icon = d > 0 ? TrendingUp : TrendingDown;
  return (
    <span className={`inline-flex items-center gap-1 text-[11px] mono font-semibold ${bad ? "text-red-400" : "text-emerald-400"}`}>
      <Icon className="w-3 h-3" />{d > 0 ? "+" : ""}{format(d)}
    </span>
  );
}

function ChevronRight(props) { return <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><polyline points="9 18 15 12 9 6" /></svg>; }

function ClassBadgeInline({ cls }) {
  const m = CLASSIFICATIONS[cls]; if (!m) return <span>{cls}</span>;
  return <span className={`inline-flex items-center px-2 py-0.5 rounded-md border text-[11px] font-semibold ${m.bg} ${m.text} ${m.border}`}>
    <span className="badge-dot" style={{ background: m.color }} />{m.short}
  </span>;
}

function Side({ c, label, labelSuffix = "" }) {
  if (!c) return null;
  const band = priorityBand(c.review_priority);
  return (
    <div className="card-surface-elevated p-5" data-testid={`compare-side-${label}`}>
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="text-[10px] uppercase tracking-widest text-cyan-400 font-semibold">Candidate {label}{labelSuffix && <span className="text-blue-400"> · {labelSuffix}</span>}</div>
          <div className="mono text-xl font-bold text-white">{c.candidate_id}</div>
        </div>
        <ClassBadgeInline cls={c.predicted_class} />
      </div>
      <div className="space-y-2 text-sm">
        {[
          { k: "Recommendation", v: <ClassBadgeInline cls={c.predicted_class} /> },
          { k: "Review Priority", v: <span className="mono font-semibold" style={{ color: band.color }}>{c.review_priority.toFixed(2)} · {band.label}</span> },
          { k: "Confidence", v: <span className="mono font-semibold text-white">{(c.confidence*100).toFixed(1)}%</span> },
          { k: "P(Warranted)", v: <span className="mono text-red-300">{(c.prob_review_warranted*100).toFixed(1)}%</span> },
          { k: "P(Not Warranted)", v: <span className="mono text-emerald-300">{(c.prob_review_not_warranted*100).toFixed(1)}%</span> },
          { k: "P(Insufficient)", v: <span className="mono text-amber-300">{(c.prob_insufficient_information*100).toFixed(1)}%</span> },
          { k: "Evidence Strength", v: <span className="text-slate-200 font-medium">{c.evidence_strength}</span> },
          { k: "Evidence Records", v: <span className="mono text-slate-300">{Object.values(c.evidence_counts).reduce((a, b) => a + b, 0)}</span> },
          { k: "New Evidence (T1)", v: <span className="mono text-blue-300">{c.new_evidence_count}</span> },
          { k: "Primary State", v: <span className="mono text-slate-200">{c.primary_state}</span> },
          { k: "Agreement (T0 vs T1)", v: <span className={c.agreement === "Agree" ? "text-emerald-300" : "text-orange-300"}>{c.agreement}</span> },
        ].map((r, i) => (
          <div key={i} className="flex items-center justify-between py-1.5 border-b border-[#151B25] last:border-b-0">
            <span className="text-slate-500 text-xs">{r.k}</span>
            <span>{r.v}</span>
          </div>
        ))}
      </div>
      <div className="mt-4 pt-4 border-t border-[#1E2633]">
        <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-2">T0 Snapshot</div>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div><div className="text-slate-500">Class</div><ClassBadgeInline cls={c.previous.predicted_class} /></div>
          <div><div className="text-slate-500">Priority</div><div className="mono font-semibold text-white">{c.previous.review_priority.toFixed(2)}</div></div>
          <div><div className="text-slate-500">Conf</div><div className="mono font-semibold text-white">{(c.previous.confidence*100).toFixed(0)}%</div></div>
        </div>
      </div>
      <div className="mt-4 pt-4 border-t border-[#1E2633]">
        <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold mb-2">Reviewer Tags</div>
        <div className="flex flex-wrap gap-1">
          {(c.reviewer_tags || []).length === 0 && <span className="text-xs text-slate-500">No tags</span>}
          {(c.reviewer_tags || []).map((t) => (
            <span key={t} className="inline-flex items-center px-2 py-0.5 rounded border border-cyan-500/30 bg-cyan-500/10 text-cyan-300 text-[11px]">{t}</span>
          ))}
        </div>
      </div>
      <Link to={`/case/${c.candidate_id}`} className="mt-4 inline-flex items-center gap-1.5 text-xs font-semibold text-blue-400 hover:text-blue-300">
        Open full investigation <ArrowRight className="w-3.5 h-3.5" />
      </Link>
    </div>
  );
}

export default function Compare() {
  const [sp, setSp] = useSearchParams();
  const [mode, setMode] = useState(sp.get("mode") === "phases" ? "phases" : "cases");
  const [a, setA] = useState(null);
  const [b, setB] = useState(null);
  const [singleCase, setSingleCase] = useState(null); // for phases mode
  const [pair, setPair] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const ida = sp.get("a"); const idb = sp.get("b");
    if (ida && !a) fetchCases({ q: ida, limit: 1 }).then(d => d.cases[0] && setA(d.cases[0]));
    if (idb && !b) fetchCases({ q: idb, limit: 1 }).then(d => d.cases[0] && setB(d.cases[0]));
  }, []); // eslint-disable-line

  useEffect(() => {
    const p = new URLSearchParams();
    if (mode === "phases") p.set("mode", "phases");
    if (a) p.set("a", a.candidate_id); if (b) p.set("b", b.candidate_id);
    setSp(p, { replace: true });
    if (mode === "cases" && a && b) {
      setLoading(true); compareCases(a.candidate_id, b.candidate_id).then(setPair).finally(() => setLoading(false));
    } else if (mode === "phases" && a) {
      setLoading(true);
      fetchCase(a.candidate_id).then((full) => {
        const t0Side = { ...full, ...full.t0, phase: "T0" };
        const t1Side = { ...full, ...full.t1, phase: "T1" };
        setSingleCase(full);
        setPair({ a: t0Side, b: t1Side });
      }).finally(() => setLoading(false));
    } else setPair(null);
  }, [a, b, mode]); // eslint-disable-line

  const swap = () => { setA(b); setB(a); };
  const changes = pair ? [
    pair.a.predicted_class !== pair.b.predicted_class && "Recommendation differs",
    Math.abs(pair.a.review_priority - pair.b.review_priority) > 0.15 && "Review priority differs materially",
    Math.abs(pair.a.confidence - pair.b.confidence) > 0.15 && "Confidence differs materially",
    pair.a.evidence_strength !== pair.b.evidence_strength && "Evidence strength differs",
    Math.abs(pair.a.new_evidence_count - pair.b.new_evidence_count) >= 3 && "Significant difference in new T1 evidence",
    pair.a.agreement !== pair.b.agreement && "Phase agreement differs",
  ].filter(Boolean) : [];

  return (
    <div className="space-y-5" data-testid="compare-page">
      <div>
        <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Investigation</div>
        <h1 className="font-display text-2xl font-bold text-white flex items-center gap-2">
          <GitCompare className="w-5 h-5 text-cyan-400" /> Case Comparison
        </h1>
        <p className="text-sm text-slate-400 mt-1">Select two candidates to compare model outputs, evidence, and T0→T1 changes side by side.</p>
      </div>

      {/* Mode toggle */}
      <div className="inline-flex items-center bg-[#121821] border border-[#232C3B] rounded-md p-0.5" data-testid="compare-mode">
        <button onClick={() => setMode("cases")} data-testid="mode-cases"
          className={`px-3 py-1.5 text-xs font-semibold rounded transition-colors ${mode === "cases" ? "bg-cyan-600 text-white" : "text-slate-400 hover:text-white"}`}>
          Two Candidates
        </button>
        <button onClick={() => setMode("phases")} data-testid="mode-phases"
          className={`px-3 py-1.5 text-xs font-semibold rounded transition-colors ${mode === "phases" ? "bg-cyan-600 text-white" : "text-slate-400 hover:text-white"}`}>
          Same Candidate — T0 vs T1
        </button>
      </div>

      <div className={`grid grid-cols-1 ${mode === "cases" ? "lg:grid-cols-2" : "lg:grid-cols-1"} gap-5`}>
        <CasePicker label={mode === "phases" ? "Candidate" : "Candidate A"} value={a} onSelect={setA} />
        {mode === "cases" && <CasePicker label="Candidate B" value={b} onSelect={setB} />}
      </div>

      {mode === "cases" && a && b && (
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={swap} data-testid="swap-cases"
            className="bg-[#121821] border-[#232C3B] text-slate-300 hover:bg-[#1B222E] hover:text-white">Swap A ↔ B</Button>
          {loading && <span className="text-xs text-slate-500">Loading comparison…</span>}
        </div>
      )}
      {mode === "phases" && loading && <span className="text-xs text-slate-500">Loading T0 vs T1…</span>}

      {pair && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <Side c={pair.a} label={mode === "phases" ? "T0" : "A"} labelSuffix={mode === "phases" ? "Initial" : ""} />
            <Side c={pair.b} label={mode === "phases" ? "T1" : "B"} labelSuffix={mode === "phases" ? "Updated" : ""} />
          </div>

          <div className="card-surface-elevated p-5" data-testid="compare-deltas">
            <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Delta</div>
            <h3 className="font-display text-base font-semibold text-white mb-4">A vs B — What Differs</h3>
            <div className="rounded-lg border border-[#1E2633] overflow-hidden">
              <div className="grid grid-cols-12 bg-[#0F141C] border-b border-[#1E2633] px-4 py-2 text-[10px] uppercase tracking-widest font-semibold text-slate-500">
                <div className="col-span-4">Attribute</div><div className="col-span-3">Candidate A</div>
                <div className="col-span-3">Candidate B</div><div className="col-span-2 text-right">Δ (B − A)</div>
              </div>
              {[
                { k: "Review Priority", va: pair.a.review_priority, vb: pair.b.review_priority, fmt: (v) => v.toFixed(2), higherIsWorse: true },
                { k: "Confidence", va: pair.a.confidence, vb: pair.b.confidence, fmt: (v) => `${(v*100).toFixed(1)}%`, higherIsWorse: false },
                { k: "P(Warranted)", va: pair.a.prob_review_warranted, vb: pair.b.prob_review_warranted, fmt: (v) => `${(v*100).toFixed(0)}%`, higherIsWorse: true },
                { k: "P(Not Warranted)", va: pair.a.prob_review_not_warranted, vb: pair.b.prob_review_not_warranted, fmt: (v) => `${(v*100).toFixed(0)}%`, higherIsWorse: false },
                { k: "P(Insufficient)", va: pair.a.prob_insufficient_information, vb: pair.b.prob_insufficient_information, fmt: (v) => `${(v*100).toFixed(0)}%`, higherIsWorse: true },
                { k: "New Evidence (T1)", va: pair.a.new_evidence_count, vb: pair.b.new_evidence_count, fmt: (v) => `${v}`, higherIsWorse: false },
              ].map((row, i) => (
                <div key={i} className="grid grid-cols-12 px-4 py-2.5 border-b border-[#151B25] last:border-b-0 items-center text-sm">
                  <div className="col-span-4 text-slate-400">{row.k}</div>
                  <div className="col-span-3 mono text-slate-300">{row.fmt(row.va)}</div>
                  <div className="col-span-3 mono text-white font-semibold">{row.fmt(row.vb)}</div>
                  <div className="col-span-2 text-right"><DeltaCell prev={row.va} cur={row.vb} format={row.fmt} higherIsWorse={row.higherIsWorse} /></div>
                </div>
              ))}
              <div className="grid grid-cols-12 px-4 py-2.5 border-b border-[#151B25] items-center text-sm">
                <div className="col-span-4 text-slate-400">Recommendation</div>
                <div className="col-span-3"><ClassBadgeInline cls={pair.a.predicted_class} /></div>
                <div className="col-span-3"><ClassBadgeInline cls={pair.b.predicted_class} /></div>
                <div className="col-span-2 text-right text-[11px] font-semibold">
                  {pair.a.predicted_class === pair.b.predicted_class
                    ? <span className="text-slate-500">Same</span>
                    : <span className="text-amber-400">Differs</span>}
                </div>
              </div>
              <div className="grid grid-cols-12 px-4 py-2.5 items-center text-sm">
                <div className="col-span-4 text-slate-400">Evidence Strength</div>
                <div className="col-span-3 text-slate-300 font-medium">{pair.a.evidence_strength}</div>
                <div className="col-span-3 text-white font-medium">{pair.b.evidence_strength}</div>
                <div className="col-span-2 text-right text-[11px] font-semibold">
                  {pair.a.evidence_strength === pair.b.evidence_strength ? <span className="text-slate-500">Same</span> : <span className="text-amber-400">Differs</span>}
                </div>
              </div>
            </div>

            {changes.length > 0 && (
              <div className="mt-4 p-3 rounded-md bg-cyan-500/5 border border-cyan-500/30">
                <div className="text-[10px] uppercase tracking-widest text-cyan-400 font-semibold mb-2 flex items-center gap-1.5">
                  <Info className="w-3.5 h-3.5" /> Key differences
                </div>
                <ul className="space-y-1">
                  {changes.map((c, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-slate-300">
                      <div className="w-1.5 h-1.5 rounded-full mt-1.5 bg-cyan-400" />
                      <span>{c}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
