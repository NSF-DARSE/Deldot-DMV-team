import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search, X, Loader2, MapPin, IdCard, Car, Briefcase, Radio, Layers, ArrowRight } from "lucide-react";
import { searchEvidence } from "../lib/api";

const SOURCE_META = {
  address: { label: "Address", icon: MapPin, color: "#3B82F6" },
  credential: { label: "Credential", icon: IdCard, color: "#8B5CF6" },
  vehicle_title: { label: "Vehicle Title", icon: Car, color: "#06B6D4" },
  work: { label: "Work", icon: Briefcase, color: "#F59E0B" },
  external: { label: "External", icon: Radio, color: "#EC4899" },
};

const CHIPS = [
  { key: "all", label: "All Evidence", icon: Layers, color: "#22D3EE" },
  { key: "address", label: "Address", icon: MapPin, color: "#3B82F6" },
  { key: "credential", label: "Credential", icon: IdCard, color: "#8B5CF6" },
  { key: "vehicle_title", label: "Vehicle Title", icon: Car, color: "#06B6D4" },
  { key: "work", label: "Work", icon: Briefcase, color: "#F59E0B" },
  { key: "external", label: "External", icon: Radio, color: "#EC4899" },
];

// Highlight matching substrings (case-insensitive) inside text
function Highlight({ text, query }) {
  if (!text) return <>{text}</>;
  if (!query || query.length < 2) return <>{text}</>;
  const t = String(text);
  const q = String(query);
  const parts = [];
  const lower = t.toLowerCase();
  const qLower = q.toLowerCase();
  let idx = 0;
  while (idx < t.length) {
    const found = lower.indexOf(qLower, idx);
    if (found === -1) { parts.push(t.slice(idx)); break; }
    if (found > idx) parts.push(t.slice(idx, found));
    parts.push(<mark key={found} className="bg-cyan-400/25 text-cyan-100 rounded px-0.5">{t.slice(found, found + q.length)}</mark>);
    idx = found + q.length;
  }
  return <>{parts}</>;
}

function humanSnippet(r) {
  // Build a readable snippet per source kind
  const parts = [];
  if (r.state) parts.push(r.state === "DE" ? "Delaware" : r.state);
  if (r.event_type) parts.push(r.event_type.replace(/_/g, " "));
  if (r.source_record_id) parts.push(r.source_record_id);
  return parts.join(" · ");
}

function relativeDate(dateStr) {
  if (!dateStr) return "";
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return dateStr;
  const days = Math.max(0, Math.floor((Date.now() - d.getTime()) / (1000 * 60 * 60 * 24)));
  if (days < 1) return "today";
  if (days < 30) return `updated ${days} day${days === 1 ? "" : "s"} ago`;
  if (days < 365) return `updated ${Math.floor(days / 30)} mo ago`;
  return `updated ${Math.floor(days / 365)} yr ago`;
}

export default function EvidenceSearch() {
  const navigate = useNavigate();
  const [q, setQ] = useState("");
  const [debounced, setDebounced] = useState("");
  const [chip, setChip] = useState("all");
  const [results, setResults] = useState({ records: [], total_matches: 0, candidates_matched: 0 });
  const [loading, setLoading] = useState(false);
  const [focused, setFocused] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setDebounced(q.trim()), 220);
    return () => clearTimeout(t);
  }, [q]);

  useEffect(() => {
    const active = debounced.length >= 2 || chip !== "all";
    if (!active) { setResults({ records: [], total_matches: 0, candidates_matched: 0 }); return; }
    let cancelled = false;
    setLoading(true);
    const params = { limit: 200 };
    if (debounced) params.q = debounced;
    if (chip !== "all") params.source = chip;
    searchEvidence(params).then((r) => { if (!cancelled) setResults(r); }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [debounced, chip]);

  const active = debounced.length >= 2 || chip !== "all";

  // Group records by candidate, then by source kind
  const grouped = useMemo(() => {
    const byCand = new Map();
    for (const r of results.records) {
      if (!byCand.has(r.candidate_id)) byCand.set(r.candidate_id, { candidate_id: r.candidate_id, sources: {}, sample: r });
      const g = byCand.get(r.candidate_id);
      if (!g.sources[r.kind]) g.sources[r.kind] = [];
      g.sources[r.kind].push(r);
    }
    return [...byCand.values()];
  }, [results]);

  const openCase = (candidateId, matchedKinds) => {
    const highlight = matchedKinds[0] || "";
    const url = highlight ? `/case/${candidateId}?highlight=${highlight}&q=${encodeURIComponent(debounced)}` : `/case/${candidateId}`;
    navigate(url);
  };

  return (
    <div className="card-surface-elevated p-5" data-testid="evidence-search">
      {/* Header + search bar */}
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Investigation</div>
          <h3 className="font-display text-lg font-semibold text-white flex items-center gap-2">
            <Search className="w-4 h-4 text-cyan-400" /> Evidence Search
          </h3>
        </div>
        <div className="text-xs text-slate-500 hidden md:block">
          Full-text lookup across <span className="mono text-white font-semibold">216,000</span> evidence records
        </div>
      </div>

      <div className={`relative rounded-lg transition-all duration-200 ${
        active || focused ? "ring-1 ring-cyan-500/40 shadow-[0_0_0_1px_rgba(34,211,238,0.15),0_10px_40px_-15px_rgba(34,211,238,0.25)]" : "ring-1 ring-transparent"
      }`}>
        <Search className="w-5 h-5 text-slate-400 absolute left-4 top-1/2 -translate-y-1/2 pointer-events-none" />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          data-testid="evidence-search-input"
          placeholder="Search by Candidate ID, address, plate, evidence type, state, or keyword..."
          className="w-full h-14 pl-12 pr-24 rounded-lg bg-[#0F141C] border border-[#1E2633] text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500/50 text-[15px] font-medium"
        />
        <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
          {loading && <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" data-testid="evidence-loading" />}
          {q && (
            <button onClick={() => setQ("")} data-testid="evidence-clear"
              className="w-7 h-7 rounded-md flex items-center justify-center text-slate-400 hover:text-white hover:bg-[#1B222E] transition-colors">
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Quick filter chips */}
      <div className="flex items-center gap-2 mt-3 flex-wrap" data-testid="evidence-chips">
        {CHIPS.map((c) => {
          const Icon = c.icon; const selected = chip === c.key;
          return (
            <button key={c.key} onClick={() => setChip(c.key)} data-testid={`evidence-chip-${c.key}`}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
                selected
                  ? "text-white"
                  : "text-slate-400 hover:text-white border-[#232C3B] bg-[#0F141C] hover:bg-[#1B222E]"
              }`}
              style={selected ? { background: `${c.color}22`, borderColor: `${c.color}66`, color: c.color } : undefined}>
              <Icon className="w-3.5 h-3.5" /> {c.label}
            </button>
          );
        })}
      </div>

      {/* Summary */}
      {active && (
        <div className="mt-4 text-xs text-slate-400" data-testid="evidence-summary">
          {loading ? "Searching…" : (
            <>
              <span className="mono font-semibold text-white">{results.candidates_matched.toLocaleString()}</span> matching case{results.candidates_matched === 1 ? "" : "s"}
              {debounced && <> found for <span className="text-cyan-300 font-semibold">&ldquo;{debounced}&rdquo;</span></>}
              {chip !== "all" && <> in <span className="text-cyan-300 font-semibold">{CHIPS.find(c => c.key === chip)?.label}</span></>}
              {results.total_matches >= 200 && <span className="text-slate-500"> · showing first 200 records</span>}
            </>
          )}
        </div>
      )}

      {/* Results */}
      {active && !loading && grouped.length === 0 && (
        <div className="mt-4 text-center py-14 rounded-lg border border-[#1E2633] bg-[#0F141C]" data-testid="evidence-empty">
          <Search className="w-8 h-8 text-slate-600 mx-auto mb-2" />
          <div className="text-sm font-semibold text-white">No matching cases found</div>
          <div className="text-xs text-slate-500 mt-1">Try a different keyword, or clear the search to browse all cases.</div>
          <button onClick={() => { setQ(""); setChip("all"); }} data-testid="evidence-clear-empty"
            className="mt-4 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-cyan-500/15 border border-cyan-500/40 text-cyan-300 text-xs font-semibold hover:bg-cyan-500/25 transition-colors">
            <X className="w-3.5 h-3.5" /> Clear search & return to all cases
          </button>
        </div>
      )}

      {active && grouped.length > 0 && (
        <div className="mt-4 space-y-2" data-testid="evidence-results">
          {grouped.slice(0, 50).map((g) => {
            const kinds = Object.keys(g.sources);
            // Pick the "best" snippet: first record whose text most closely matches the query
            const best = kinds.map(k => g.sources[k][0]).find(Boolean) || g.sample;
            return (
              <button key={g.candidate_id} onClick={() => openCase(g.candidate_id, kinds)} data-testid={`evidence-result-${g.candidate_id}`}
                className="w-full text-left p-3.5 rounded-lg border border-[#232C3B] bg-[#0F141C] hover:border-cyan-500/40 hover:bg-[#141B26] transition-all">
                <div className="flex items-center justify-between gap-3 flex-wrap">
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="mono text-sm font-bold text-white truncate">
                      <Highlight text={g.candidate_id} query={debounced} />
                    </div>
                    <div className="flex items-center gap-1.5 flex-wrap">
                      {kinds.map(k => {
                        const meta = SOURCE_META[k]; const Icon = meta.icon;
                        return (
                          <span key={k} data-testid={`match-badge-${k}`}
                            className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold border"
                            style={{ background: `${meta.color}18`, borderColor: `${meta.color}44`, color: meta.color }}>
                            <Icon className="w-2.5 h-2.5" /> Matched: {meta.label}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                  <ArrowRight className="w-4 h-4 text-slate-500" />
                </div>
                {/* Matching snippet(s) */}
                <div className="mt-2 pl-1 space-y-1">
                  {kinds.slice(0, 3).map(k => {
                    const rec = g.sources[k][0]; const meta = SOURCE_META[k];
                    return (
                      <div key={k} className="text-[12px] text-slate-400 leading-relaxed">
                        <span className="mono font-semibold" style={{ color: meta.color }}>{meta.label} —</span>{" "}
                        <span>
                          <Highlight text={humanSnippet(rec)} query={debounced} />
                          {rec.date && <span className="text-slate-500"> — {relativeDate(rec.date)}</span>}
                        </span>
                      </div>
                    );
                  })}
                  {kinds.length > 3 && <div className="text-[11px] text-slate-500">…and {kinds.length - 3} more matched source{kinds.length - 3 === 1 ? "" : "s"}</div>}
                </div>
              </button>
            );
          })}
          {grouped.length > 50 && (
            <div className="text-center text-xs text-slate-500 py-2">Showing first 50 candidates · refine your search for more precise results</div>
          )}
        </div>
      )}
    </div>
  );
}
