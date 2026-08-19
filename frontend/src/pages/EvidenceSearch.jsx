import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Search, MapPin, IdCard, Car, Briefcase, Radio, Filter, ArrowRight } from "lucide-react";
import { searchEvidence } from "../lib/api";
import { Input } from "../components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";

const SOURCE_META = {
  address: { label: "Address", icon: MapPin, color: "#3B82F6" },
  credential: { label: "Credential", icon: IdCard, color: "#8B5CF6" },
  vehicle_title: { label: "Vehicle Title", icon: Car, color: "#06B6D4" },
  work: { label: "Work", icon: Briefcase, color: "#F59E0B" },
  external: { label: "External", icon: Radio, color: "#EC4899" },
};

const US_STATES = ["all", "DE", "PA", "MD", "NJ", "VA", "NY", "OH", "WV", "NC", "FL", "TX"];

export default function EvidenceSearch() {
  const [q, setQ] = useState("");
  const [source, setSource] = useState("all");
  const [state, setState] = useState("all");
  const [results, setResults] = useState({ records: [], total_matches: 0, candidates_matched: 0 });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const t = setTimeout(async () => {
      if (!q && source === "all" && state === "all") { setResults({ records: [], total_matches: 0, candidates_matched: 0 }); return; }
      setLoading(true);
      const params = { limit: 100 };
      if (q) params.q = q;
      if (source !== "all") params.source = source;
      if (state !== "all") params.state = state;
      const r = await searchEvidence(params);
      setResults(r);
      setLoading(false);
    }, 200);
    return () => clearTimeout(t);
  }, [q, source, state]);

  return (
    <div className="space-y-5" data-testid="evidence-search-page">
      <div>
        <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Investigation</div>
        <h1 className="font-display text-2xl font-bold text-white flex items-center gap-2">
          <Search className="w-5 h-5 text-cyan-400" /> Evidence Search
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Search across <span className="text-white font-semibold">216,000</span> evidence records — candidate IDs,
          source record IDs (LIC-, ADR-, TTL-, WRK-, EXT-), vehicle references, states, and event types.
        </p>
      </div>

      <div className="card-surface p-4">
        <div className="flex items-center gap-2 mb-3 text-xs uppercase tracking-widest text-slate-500 font-semibold">
          <Filter className="w-3.5 h-3.5" /> Search
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div className="md:col-span-2 relative">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <Input value={q} onChange={(e) => setQ(e.target.value)}
              placeholder="e.g. CAN-B6WV6HQ3JW, LIC-874Z2AAGZU, VH-49369, DE"
              className="pl-9 h-10 bg-[#0F141C] border-[#1E2633] text-slate-200 placeholder:text-slate-500" data-testid="evidence-search-input" />
          </div>
          <Select value={source} onValueChange={setSource}>
            <SelectTrigger className="h-10 bg-[#0F141C] border-[#1E2633] text-slate-200" data-testid="evidence-source-filter"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Sources</SelectItem>
              {Object.entries(SOURCE_META).map(([k, v]) => <SelectItem key={k} value={k}>{v.label}</SelectItem>)}
            </SelectContent>
          </Select>
          <Select value={state} onValueChange={setState}>
            <SelectTrigger className="h-10 bg-[#0F141C] border-[#1E2633] text-slate-200" data-testid="evidence-state-filter"><SelectValue /></SelectTrigger>
            <SelectContent>{US_STATES.map(s => <SelectItem key={s} value={s}>{s === "all" ? "All States" : s}</SelectItem>)}</SelectContent>
          </Select>
        </div>
      </div>

      <div className="text-xs text-slate-400" data-testid="evidence-counts">
        {loading ? "Searching…" : (
          <>
            <span className="mono font-semibold text-white">{results.total_matches.toLocaleString()}</span> record{results.total_matches===1?"":"s"} matched
            {results.candidates_matched > 0 && (
              <> · <span className="mono font-semibold text-cyan-300">{results.candidates_matched.toLocaleString()}</span> candidate{results.candidates_matched===1?"":"s"}</>
            )}
            {results.total_matches >= 100 && <span className="text-slate-500"> · showing first 100</span>}
          </>
        )}
      </div>

      <div className="card-surface overflow-hidden" data-testid="evidence-results">
        <div className="grid grid-cols-12 bg-[#0F141C] border-b border-[#1E2633] px-4 py-2 text-[10px] uppercase tracking-widest font-semibold text-slate-500">
          <div className="col-span-2">Candidate</div>
          <div className="col-span-2">Source</div>
          <div className="col-span-3">Source Record</div>
          <div className="col-span-1">State</div>
          <div className="col-span-2">Event</div>
          <div className="col-span-1">Date</div>
          <div className="col-span-1 text-right">Open</div>
        </div>
        {results.records.length === 0 && !loading && (
          <div className="text-center py-14 text-slate-500" data-testid="empty-search">
            {q || source !== "all" || state !== "all"
              ? "No evidence records matched."
              : "Enter a search term or filter to explore evidence records."}
          </div>
        )}
        {results.records.map((r, i) => {
          const meta = SOURCE_META[r.kind];
          const Icon = meta?.icon || Search;
          return (
            <Link key={i} to={`/case/${r.candidate_id}`}
              className="grid grid-cols-12 px-4 py-2.5 border-b border-[#151B25] last:border-b-0 items-center text-sm row-hover"
              data-testid={`evidence-row-${i}`}>
              <div className="col-span-2 mono text-xs font-semibold text-white truncate">{r.candidate_id}</div>
              <div className="col-span-2 flex items-center gap-2">
                <Icon className="w-3.5 h-3.5" style={{ color: meta?.color }} />
                <span className="text-xs text-slate-300">{meta?.label || r.kind}</span>
              </div>
              <div className="col-span-3 mono text-xs text-slate-400 truncate">
                {r.source_record_id}
                {r.vehicle_ref && <span className="ml-2 text-cyan-400">{r.vehicle_ref}</span>}
              </div>
              <div className="col-span-1">
                <span className={`mono text-[11px] font-semibold px-1.5 py-0.5 rounded border ${
                  r.state === "DE" ? "border-blue-500/30 bg-blue-500/10 text-blue-300"
                  : "border-slate-500/30 bg-slate-500/10 text-slate-300"
                }`}>{r.state}</span>
              </div>
              <div className="col-span-2 text-xs text-slate-400 truncate">{r.event_type || "—"}</div>
              <div className="col-span-1 mono text-[11px] text-slate-500">{r.date}</div>
              <div className="col-span-1 text-right"><ArrowRight className="w-3.5 h-3.5 text-slate-500 inline" /></div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
