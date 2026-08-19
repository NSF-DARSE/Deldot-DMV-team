import React, { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Search, X, RefreshCw, Download, SlidersHorizontal, Tag as TagIcon, CheckSquare, Plus } from "lucide-react";
import { fetchCases, fetchTags, fetchDataStatus, bulkAddTag, CLASSIFICATIONS, REVIEWER_STATUS_META, SUGGESTED_TAGS, exportCsvUrl } from "../lib/api";
import { toast } from "sonner";
import CaseTable from "../components/CaseTable";
import EvidenceSearch from "../components/EvidenceSearch";
import { Input } from "../components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Button } from "../components/ui/button";

export default function Queue({ presetClass, title = "Review Queue", accent = "#60A5FA" }) {
  const [sp, setSp] = useSearchParams();
  const [q, setQ] = useState(sp.get("q") || "");
  const [predicted, setPredicted] = useState(presetClass || sp.get("predicted_class") || "all");
  const [status, setStatus] = useState(sp.get("reviewer_status") || "all");
  const [band, setBand] = useState(sp.get("band") || "all");
  const [phase, setPhase] = useState(sp.get("phase") || "T1"); // T0/T1 view; "ALL" shows both
  const [evStrength, setEvStrength] = useState(sp.get("evidence_strength") || "all");
  const [tag, setTag] = useState(sp.get("tag") || "all");
  const [sortBy, setSortBy] = useState(sp.get("sort_by") || "review_priority");
  const [sortDir, setSortDir] = useState(sp.get("sort_dir") || "desc");
  const [cases, setCases] = useState([]);
  const [total, setTotal] = useState(0);
  const [tagList, setTagList] = useState([]);
  const [dataStatus, setDataStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);
  const [selection, setSelection] = useState(new Set());
  const [bulkTag, setBulkTag] = useState("");
  const [bulkOpen, setBulkOpen] = useState(false);
  const perPage = 25;

  const toggleSelect = (id) => {
    const s = new Set(selection);
    if (s.has(id)) s.delete(id); else s.add(id);
    setSelection(s);
  };
  const toggleSelectAll = (rows) => {
    const s = new Set(selection);
    const allChecked = rows.every(r => s.has(r.candidate_id));
    if (allChecked) rows.forEach(r => s.delete(r.candidate_id));
    else rows.forEach(r => s.add(r.candidate_id));
    setSelection(s);
  };
  const applyBulkTag = async (tag) => {
    const t = (tag || bulkTag || "").trim();
    if (!t || selection.size === 0) return;
    try {
      const r = await bulkAddTag([...selection], t);
      toast.success(`Tagged ${r.tagged} case${r.tagged === 1 ? "" : "s"} · ${t}`);
      setBulkTag(""); setBulkOpen(false); setSelection(new Set());
      load();
    } catch { toast.error("Bulk tag failed"); }
  };

  useEffect(() => { setPredicted(presetClass || "all"); }, [presetClass]);
  useEffect(() => { (async () => { setTagList((await fetchTags()).tags); setDataStatus(await fetchDataStatus()); })(); }, []);

  const buildParams = (phaseArg) => {
    const params = { sort_by: sortBy, sort_dir: sortDir, skip: (page - 1) * perPage, limit: perPage, phase: phaseArg };
    if (q) params.q = q;
    if (predicted !== "all") params.predicted_class = predicted;
    if (status !== "all") params.reviewer_status = status;
    if (band !== "all") params.priority_band = band;
    if (evStrength !== "all") params.evidence_strength = evStrength;
    if (tag !== "all") params.tag = tag;
    return params;
  };

  const load = async () => {
    setLoading(true);
    if (phase === "ALL") {
      // Fetch both phases and interleave
      const [t0, t1] = await Promise.all([
        fetchCases(buildParams("T0")),
        fetchCases(buildParams("T1")),
      ]);
      // combined view — merge on candidate priority
      const merged = [...t0.cases, ...t1.cases].sort((a, b) => (sortDir === "desc" ? b[sortBy] - a[sortBy] : a[sortBy] - b[sortBy]));
      setCases(merged);
      setTotal(t0.total + t1.total);
    } else {
      const data = await fetchCases(buildParams(phase));
      setCases(data.cases);
      setTotal(data.total);
    }
    setLoading(false);
  };

  useEffect(() => {
    load();
    const p = new URLSearchParams();
    if (q) p.set("q", q);
    if (!presetClass && predicted !== "all") p.set("predicted_class", predicted);
    if (status !== "all") p.set("reviewer_status", status);
    if (band !== "all") p.set("band", band);
    if (phase !== "T1") p.set("phase", phase);
    if (evStrength !== "all") p.set("evidence_strength", evStrength);
    if (tag !== "all") p.set("tag", tag);
    p.set("sort_by", sortBy); p.set("sort_dir", sortDir);
    setSp(p, { replace: true });
  }, [q, predicted, status, band, phase, evStrength, tag, sortBy, sortDir, page]); // eslint-disable-line

  useEffect(() => { setPage(1); }, [q, predicted, status, band, phase, evStrength, tag, sortBy, sortDir]);

  const onSort = (col) => {
    if (sortBy === col) setSortDir(sortDir === "asc" ? "desc" : "asc");
    else { setSortBy(col); setSortDir("desc"); }
  };
  const clear = () => { setQ(""); if (!presetClass) setPredicted("all"); setStatus("all"); setBand("all"); setEvStrength("all"); setTag("all"); };
  const active = (q ? 1 : 0) + ((!presetClass && predicted !== "all") ? 1 : 0) + (status !== "all" ? 1 : 0) + (band !== "all" ? 1 : 0) + (evStrength !== "all" ? 1 : 0) + (tag !== "all" ? 1 : 0);
  const pageCount = Math.max(1, Math.ceil(total / perPage));

  return (
    <div className="space-y-5" data-testid="queue-page">
      <div className="flex items-end justify-between flex-wrap gap-3">
        <div>
          <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Operations</div>
          <h1 className="font-display text-2xl font-bold" style={{ color: accent }}>{title}</h1>
          <p className="text-sm text-slate-400 mt-1" data-testid="queue-counts">
            {loading ? "Loading…" : (
              <>
                <span className="text-white font-semibold">{total.toLocaleString()}</span> case{total === 1 ? "" : "s"}
                {dataStatus && !presetClass && (
                  <span className="text-slate-500"> · Warranted <span className="text-red-300 font-semibold">{dataStatus.by_class_t1.review_warranted?.toLocaleString() || 0}</span>
                    {" "}· Not Warranted <span className="text-emerald-300 font-semibold">{dataStatus.by_class_t1.review_not_warranted?.toLocaleString() || 0}</span>
                    {" "}· Insufficient <span className="text-amber-300 font-semibold">{dataStatus.by_class_t1.insufficient_information?.toLocaleString() || 0}</span>
                  </span>
                )}
                {active > 0 && ` · ${active} filter${active > 1 ? "s" : ""}`}
              </>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Phase tabs */}
          <div className="flex items-center gap-0.5 bg-[#121821] border border-[#232C3B] rounded-md p-0.5" data-testid="phase-tabs">
            {["ALL", "T0", "T1"].map((pv) => (
              <button key={pv} onClick={() => setPhase(pv)} data-testid={`phase-tab-${pv}`}
                className={`px-3 py-1.5 text-xs font-semibold rounded transition-colors ${phase === pv ? "bg-blue-600 text-white" : "text-slate-400 hover:text-white"}`}>
                {pv === "ALL" ? "All Cases" : `${pv} Cases`}
              </button>
            ))}
          </div>
          <a href={exportCsvUrl()} data-testid="export-csv" className="inline-flex items-center gap-1.5 px-3 h-9 rounded-md bg-[#121821] border border-[#232C3B] text-sm font-medium text-slate-300 hover:bg-[#1B222E] hover:text-white transition-colors">
            <Download className="w-4 h-4" /> Export CSV
          </a>
          <Button variant="outline" size="sm" onClick={load} data-testid="refresh-queue" className="bg-[#121821] border-[#232C3B] text-slate-300 hover:bg-[#1B222E] hover:text-white">
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? "animate-spin" : ""}`} /> Refresh
          </Button>
        </div>
      </div>

      {/* Prominent Evidence Search — directly below the page header */}
      <EvidenceSearch />

      <div className="card-surface p-4">
        <div className="flex items-center gap-2 mb-3 text-xs uppercase tracking-widest text-slate-500 font-semibold">
          <SlidersHorizontal className="w-3.5 h-3.5" /> Filters
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-3">
          <div className="lg:col-span-2 relative">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search candidate ID (CAN-...)"
              className="pl-9 h-10 bg-[#0F141C] border-[#1E2633] text-slate-200 placeholder:text-slate-500" data-testid="filter-search" />
          </div>
          {!presetClass && (
            <Select value={predicted} onValueChange={setPredicted}>
              <SelectTrigger className="h-10 bg-[#0F141C] border-[#1E2633] text-slate-200" data-testid="filter-class"><SelectValue placeholder="Recommendation" /></SelectTrigger>
              <SelectContent><SelectItem value="all">All Recommendations</SelectItem>
                {Object.entries(CLASSIFICATIONS).map(([k, v]) => <SelectItem key={k} value={k}>{v.label}</SelectItem>)}
              </SelectContent>
            </Select>
          )}
          <Select value={band} onValueChange={setBand}>
            <SelectTrigger className="h-10 bg-[#0F141C] border-[#1E2633] text-slate-200" data-testid="filter-band"><SelectValue placeholder="Priority" /></SelectTrigger>
            <SelectContent><SelectItem value="all">All Priorities</SelectItem>
              <SelectItem value="critical">Critical (≥ 0.85)</SelectItem><SelectItem value="high">High (0.70 – 0.84)</SelectItem>
              <SelectItem value="medium">Medium (0.40 – 0.69)</SelectItem><SelectItem value="low">Low (&lt; 0.40)</SelectItem>
            </SelectContent>
          </Select>
          <Select value={evStrength} onValueChange={setEvStrength}>
            <SelectTrigger className="h-10 bg-[#0F141C] border-[#1E2633] text-slate-200" data-testid="filter-evstrength"><SelectValue placeholder="Evidence Strength" /></SelectTrigger>
            <SelectContent><SelectItem value="all">All Strengths</SelectItem>
              <SelectItem value="Strong">Strong</SelectItem><SelectItem value="Moderate">Moderate</SelectItem><SelectItem value="Weak">Weak</SelectItem>
            </SelectContent>
          </Select>
          <Select value={status} onValueChange={setStatus}>
            <SelectTrigger className="h-10 bg-[#0F141C] border-[#1E2633] text-slate-200" data-testid="filter-status"><SelectValue placeholder="Reviewer Status" /></SelectTrigger>
            <SelectContent><SelectItem value="all">All Statuses</SelectItem>
              {Object.entries(REVIEWER_STATUS_META).map(([k, v]) => <SelectItem key={k} value={k}>{v.label}</SelectItem>)}
            </SelectContent>
          </Select>
          <Select value={tag} onValueChange={setTag}>
            <SelectTrigger className="h-10 bg-[#0F141C] border-[#1E2633] text-slate-200" data-testid="filter-tag">
              <TagIcon className="w-3.5 h-3.5 text-slate-500 mr-1" /><SelectValue placeholder="Tag" />
            </SelectTrigger>
            <SelectContent><SelectItem value="all">All Tags</SelectItem>
              {tagList.map((t) => <SelectItem key={t.tag} value={t.tag}>{t.tag} ({t.count})</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        {active > 0 && (
          <div className="mt-3">
            <Button variant="ghost" size="sm" onClick={clear} data-testid="clear-filters" className="text-slate-400 hover:text-white">
              <X className="w-3.5 h-3.5 mr-1" /> Clear filters
            </Button>
          </div>
        )}
      </div>

      {/* Bulk toolbar */}
      {selection.size > 0 && (
        <div className="card-surface-elevated p-3 flex items-center justify-between flex-wrap gap-2" data-testid="bulk-toolbar">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-md bg-cyan-500/15 border border-cyan-500/30 flex items-center justify-center">
              <CheckSquare className="w-4 h-4 text-cyan-400" />
            </div>
            <div>
              <div className="text-sm font-semibold text-white"><span className="mono">{selection.size}</span> case{selection.size===1?"":"s"} selected</div>
              <div className="text-[11px] text-slate-500">Bulk actions don&apos;t change model predictions or make enforcement decisions.</div>
            </div>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            {bulkOpen ? (
              <>
                <input value={bulkTag} onChange={(e) => setBulkTag(e.target.value)} placeholder="Enter tag…"
                  onKeyDown={(e) => e.key === "Enter" && applyBulkTag()}
                  className="h-9 px-3 rounded-md bg-[#0F141C] border border-[#1E2633] text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500/50"
                  data-testid="bulk-tag-input" />
                <Button size="sm" onClick={() => applyBulkTag()} disabled={!bulkTag.trim()} data-testid="bulk-tag-apply"
                  className="h-9 bg-cyan-600 hover:bg-cyan-500 text-white">Apply</Button>
                <Button size="sm" variant="ghost" onClick={() => { setBulkOpen(false); setBulkTag(""); }}
                  className="h-9 text-slate-400 hover:text-white">Cancel</Button>
              </>
            ) : (
              <>
                <div className="flex items-center gap-1 flex-wrap">
                  {SUGGESTED_TAGS.slice(0, 4).map(t => (
                    <button key={t} onClick={() => applyBulkTag(t)} data-testid={`bulk-suggest-${t}`}
                      className="text-[11px] px-2 py-1 rounded border border-[#232C3B] bg-[#0F141C] text-slate-400 hover:text-cyan-300 hover:border-cyan-500/40 transition-colors">
                      + {t}
                    </button>
                  ))}
                </div>
                <Button size="sm" onClick={() => setBulkOpen(true)} data-testid="bulk-tag-open"
                  className="h-9 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-300">
                  <Plus className="w-3.5 h-3.5 mr-1" /> Custom tag
                </Button>
                <Button size="sm" variant="outline" onClick={() => setSelection(new Set())} data-testid="bulk-clear"
                  className="h-9 bg-[#121821] border-[#232C3B] text-slate-300 hover:bg-[#1B222E] hover:text-white">
                  <X className="w-3.5 h-3.5 mr-1" /> Clear
                </Button>
              </>
            )}
          </div>
        </div>
      )}

      <CaseTable cases={cases} sortBy={sortBy} sortDir={sortDir} onSort={onSort} showPhase={phase === "ALL"}
        selection={selection} onToggleSelect={toggleSelect} onToggleSelectAll={toggleSelectAll} />

      {total > perPage && (
        <div className="flex items-center justify-between text-xs text-slate-500" data-testid="pagination">
          <div>Showing {(page-1)*perPage + 1}–{Math.min(page*perPage, total)} of {total.toLocaleString()}</div>
          <div className="flex items-center gap-1">
            <Button variant="outline" size="sm" disabled={page===1} onClick={() => setPage(p=>p-1)} data-testid="page-prev"
              className="h-8 bg-[#121821] border-[#232C3B] text-slate-300 disabled:opacity-40">Prev</Button>
            <span className="px-3 mono">{page} / {pageCount.toLocaleString()}</span>
            <Button variant="outline" size="sm" disabled={page>=pageCount} onClick={() => setPage(p=>p+1)} data-testid="page-next"
              className="h-8 bg-[#121821] border-[#232C3B] text-slate-300 disabled:opacity-40">Next</Button>
          </div>
        </div>
      )}
    </div>
  );
}
