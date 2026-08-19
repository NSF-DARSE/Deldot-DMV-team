import React from "react";
import { useNavigate } from "react-router-dom";
import { ArrowUpDown, Eye, Tag as TagIcon } from "lucide-react";
import { CLASSIFICATIONS, priorityBand, REVIEWER_STATUS_META } from "../lib/api";
import { Button } from "./ui/button";

function PriorityBar({ value }) {
  const band = priorityBand(value);
  return (
    <div className="flex items-center gap-2 min-w-[120px]">
      <div className="flex-1 h-1.5 rounded-full bg-[#1B222E] overflow-hidden">
        <div className="h-full gauge-fill rounded-full" style={{ width: `${value * 100}%`, background: band.color, boxShadow: `0 0 6px ${band.color}66` }} />
      </div>
      <span className="mono text-xs font-semibold text-white w-9 text-right">{value.toFixed(2)}</span>
    </div>
  );
}
function ClassBadge({ cls }) {
  const meta = CLASSIFICATIONS[cls];
  if (!meta) return <span className="text-xs text-slate-500">{cls}</span>;
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-md border text-xs font-semibold ${meta.bg} ${meta.text} ${meta.border}`} data-testid={`class-badge-${cls}`}>
      <span className="badge-dot" style={{ background: meta.color }} /> {meta.short}
    </span>
  );
}
function PriorityBadge({ value }) {
  const b = priorityBand(value);
  return <span className={`inline-flex items-center px-2 py-0.5 rounded-md border text-[11px] font-semibold ${b.bg} ${b.text} ${b.border}`}>{b.label}</span>;
}
function TagChips({ tags }) {
  if (!tags || tags.length === 0) return <span className="text-[10px] text-slate-600">—</span>;
  const shown = tags.slice(0, 2);
  return (
    <div className="flex items-center gap-1 flex-wrap max-w-[160px]">
      {shown.map((t) => (
        <span key={t} className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded border border-cyan-500/30 bg-cyan-500/10 text-cyan-300 text-[10px] font-medium">
          <TagIcon className="w-2.5 h-2.5" />{t}
        </span>
      ))}
      {tags.length > 2 && <span className="text-[10px] text-slate-500">+{tags.length - 2}</span>}
    </div>
  );
}
const strengthColor = (s) => ({ Strong: "text-emerald-300", Moderate: "text-amber-300", Weak: "text-slate-400" }[s] || "text-slate-400");
const recencyColor = (r) => ({ Recent: "text-emerald-300", Mixed: "text-amber-300", Stale: "text-slate-400" }[r] || "text-slate-400");
const agreementColor = (a) => a === "Agree" ? "text-emerald-300" : "text-orange-300";

function SortHeader({ label, k, sortBy, sortDir, onSort }) {
  return (
    <button className="inline-flex items-center gap-1 text-[10px] uppercase tracking-widest font-semibold text-slate-500 hover:text-slate-200"
      onClick={() => onSort(k)} data-testid={`sort-${k}`}>
      {label}<ArrowUpDown className="w-3 h-3" />
      {sortBy === k && <span className="text-slate-300">{sortDir === "asc" ? "↑" : "↓"}</span>}
    </button>
  );
}

export default function CaseTable({ cases, sortBy, sortDir, onSort, showPhase = true, selection, onToggleSelect, onToggleSelectAll }) {
  const navigate = useNavigate();
  const sh = (label, k) => <SortHeader label={label} k={k} sortBy={sortBy} sortDir={sortDir} onSort={onSort} />;
  const allSelected = selection && cases.length > 0 && cases.every(c => selection.has(c.candidate_id));
  return (
    <div className="card-surface overflow-hidden" data-testid="case-table">
      <div className="overflow-x-auto">
        <table className="w-full text-sm min-w-[1550px]">
          <thead className="bg-[#0F141C] border-b border-[#1E2633]">
            <tr className="text-left">
              {selection && (
                <th className="px-3 py-3 w-8">
                  <input type="checkbox" checked={allSelected} onChange={() => onToggleSelectAll?.(cases)}
                    data-testid="select-all" className="w-3.5 h-3.5 accent-cyan-500 cursor-pointer" />
                </th>
              )}
              <th className="px-4 py-3">{sh("Priority", "review_priority")}</th>
              <th className="px-4 py-3">{sh("Candidate ID", "candidate_id")}</th>
              {showPhase && <th className="px-4 py-3">{sh("Phase", "phase")}</th>}
              <th className="px-4 py-3">{sh("Recommendation", "predicted_class")}</th>
              <th className="px-4 py-3">{sh("Priority Score", "review_priority")}</th>
              <th className="px-4 py-3 text-[10px] uppercase tracking-widest font-semibold text-slate-500">P(W)</th>
              <th className="px-4 py-3 text-[10px] uppercase tracking-widest font-semibold text-slate-500">P(NW)</th>
              <th className="px-4 py-3 text-[10px] uppercase tracking-widest font-semibold text-slate-500">P(II)</th>
              <th className="px-4 py-3 text-[10px] uppercase tracking-widest font-semibold text-slate-500">Ev. Strength</th>
              <th className="px-4 py-3 text-[10px] uppercase tracking-widest font-semibold text-slate-500">Agreement</th>
              <th className="px-4 py-3 text-[10px] uppercase tracking-widest font-semibold text-slate-500">Tags</th>
              <th className="px-4 py-3 text-right text-[10px] uppercase tracking-widest font-semibold text-slate-500">Action</th>
            </tr>
          </thead>
          <tbody>
            {cases.length === 0 && (
              <tr><td colSpan={selection ? (showPhase ? 13 : 12) : (showPhase ? 12 : 11)} className="text-center py-16 text-slate-500" data-testid="empty-cases">No cases matching filters.</td></tr>
            )}
            {cases.map((c) => {
              const checked = selection?.has(c.candidate_id);
              return (
              <tr key={`${c.candidate_id}-${c.phase}`} className={`border-b border-[#151B25] last:border-b-0 row-hover cursor-pointer ${checked ? "bg-cyan-500/5" : ""}`}
                data-testid={`case-row-${c.candidate_id}`} onClick={() => navigate(`/case/${c.candidate_id}`)}>
                {selection && (
                  <td className="px-3 py-3" onClick={(e) => { e.stopPropagation(); onToggleSelect?.(c.candidate_id); }}>
                    <input type="checkbox" checked={!!checked} readOnly data-testid={`select-${c.candidate_id}`} className="w-3.5 h-3.5 accent-cyan-500 cursor-pointer" />
                  </td>
                )}
                <td className="px-4 py-3"><PriorityBadge value={c.review_priority} /></td>
                <td className="px-4 py-3 mono font-semibold text-white text-xs">{c.candidate_id}</td>
                {showPhase && <td className="px-4 py-3"><span className="mono text-xs font-semibold text-blue-300 px-2 py-0.5 rounded border border-blue-500/30 bg-blue-500/10">{c.phase}</span></td>}
                <td className="px-4 py-3"><ClassBadge cls={c.predicted_class} /></td>
                <td className="px-4 py-3"><PriorityBar value={c.review_priority} /></td>
                <td className="px-4 py-3 mono text-xs text-red-300">{(c.prob_review_warranted*100).toFixed(0)}%</td>
                <td className="px-4 py-3 mono text-xs text-emerald-300">{(c.prob_review_not_warranted*100).toFixed(0)}%</td>
                <td className="px-4 py-3 mono text-xs text-amber-300">{(c.prob_insufficient_information*100).toFixed(0)}%</td>
                <td className={`px-4 py-3 text-xs font-medium ${strengthColor(c.evidence_strength)}`}>{c.evidence_strength}</td>
                <td className={`px-4 py-3 text-xs font-medium ${agreementColor(c.agreement)}`}>{c.agreement}</td>
                <td className="px-4 py-3"><TagChips tags={c.reviewer_tags} /></td>
                <td className="px-4 py-3 text-right">
                  <Button size="sm" variant="outline" onClick={(e) => { e.stopPropagation(); navigate(`/case/${c.candidate_id}`); }}
                    data-testid={`inspect-${c.candidate_id}`} className="h-8 bg-transparent border-[#232C3B] text-slate-300 hover:bg-[#1B222E] hover:text-white">
                    <Eye className="w-3.5 h-3.5 mr-1.5" /> Inspect
                  </Button>
                </td>
              </tr>
            );})}
          </tbody>
        </table>
      </div>
    </div>
  );
}
