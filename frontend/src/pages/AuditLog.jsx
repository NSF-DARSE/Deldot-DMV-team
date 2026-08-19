import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchAudit, fetchAuditEntry, REVIEWER_STATUS_META, CLASSIFICATIONS } from "../lib/api";
import { ClipboardList, RefreshCw, X } from "lucide-react";
import { Button } from "../components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../components/ui/dialog";

export default function AuditLog() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(null);

  const load = async () => { setLoading(true); const d = await fetchAudit(); setEntries(d.entries); setLoading(false); };
  useEffect(() => { load(); }, []);

  const open = async (id) => { const d = await fetchAuditEntry(id); setSelected(d); };

  return (
    <div className="space-y-5" data-testid="audit-log-page">
      <div className="flex items-end justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Governance</div>
          <h1 className="font-display text-2xl font-bold text-white">Audit History</h1>
          <p className="text-sm text-slate-400 mt-1">Append-only log of analyst decisions and system events.</p>
        </div>
        <Button variant="outline" size="sm" onClick={load} data-testid="refresh-audit"
          className="bg-[#121821] border-[#232C3B] text-slate-300 hover:bg-[#1B222E] hover:text-white">
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? "animate-spin" : ""}`} /> Refresh
        </Button>
      </div>

      <div className="card-surface overflow-hidden">
        <div className="grid grid-cols-12 bg-[#0F141C] border-b border-[#1E2633] px-4 py-2 text-[10px] uppercase tracking-widest font-semibold text-slate-500">
          <div className="col-span-3">Timestamp</div>
          <div className="col-span-2">Candidate</div>
          <div className="col-span-1">Phase</div>
          <div className="col-span-1">Model</div>
          <div className="col-span-2">Reviewer</div>
          <div className="col-span-2">Action</div>
          <div className="col-span-1 text-right">Detail</div>
        </div>
        {entries.length === 0 && (
          <div className="text-center py-14 text-slate-500" data-testid="empty-audit">
            <ClipboardList className="w-8 h-8 mx-auto mb-2" /> No audit entries yet.
          </div>
        )}
        {entries.map((e) => (
          <div key={e.id} className="grid grid-cols-12 px-4 py-3 border-b border-[#151B25] last:border-b-0 items-center text-sm row-hover" data-testid={`audit-row-${e.id}`}>
            <div className="col-span-3 mono text-xs text-slate-400">{new Date(e.timestamp).toLocaleString()}</div>
            <div className="col-span-2 mono text-xs font-semibold text-white">
              {e.candidate_id === "BULK" || e.candidate_id === "SYSTEM"
                ? <span className="text-slate-500">{e.candidate_id}</span>
                : <Link to={`/case/${e.candidate_id}`} className="text-blue-400 hover:text-blue-300">{e.candidate_id}</Link>}
            </div>
            <div className="col-span-1"><span className="mono text-[11px] font-semibold text-blue-300 px-1.5 py-0.5 rounded border border-blue-500/30 bg-blue-500/10">{e.phase}</span></div>
            <div className="col-span-1 mono text-[11px] text-slate-500">{e.model_version}</div>
            <div className="col-span-2 mono text-xs text-slate-300">{e.reviewer_id}</div>
            <div className="col-span-2">
              <span className="text-xs mono font-medium text-slate-200 px-2 py-0.5 rounded border border-[#232C3B] bg-[#121821]">
                {REVIEWER_STATUS_META[e.to_status]?.label || e.action}
              </span>
            </div>
            <div className="col-span-1 text-right">
              <button className="text-xs text-blue-400 hover:text-blue-300 font-semibold" onClick={() => open(e.id)} data-testid={`audit-open-${e.id}`}>Open</button>
            </div>
          </div>
        ))}
      </div>

      <Dialog open={!!selected} onOpenChange={(o) => { if (!o) setSelected(null); }}>
        <DialogContent className="bg-[#121821] border-[#232C3B] text-slate-200 max-w-2xl" data-testid="audit-modal">
          {selected && (
            <>
              <DialogHeader>
                <DialogTitle className="text-white font-display flex items-center gap-2">
                  Audit Record
                  <span className="mono text-xs text-slate-500 font-normal">{selected.id.slice(0, 24)}…</span>
                </DialogTitle>
              </DialogHeader>
              <div className="space-y-4 mt-2">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div><div className="text-[10px] uppercase tracking-widest text-slate-500">Candidate</div><div className="mono font-semibold text-white">{selected.candidate_id}</div></div>
                  <div><div className="text-[10px] uppercase tracking-widest text-slate-500">Timestamp</div><div className="mono text-slate-300">{new Date(selected.timestamp).toLocaleString()}</div></div>
                  <div><div className="text-[10px] uppercase tracking-widest text-slate-500">Reviewer</div><div className="mono text-slate-300">{selected.reviewer_id}</div></div>
                  <div><div className="text-[10px] uppercase tracking-widest text-slate-500">Model</div><div className="mono text-slate-300">{selected.model_version}</div></div>
                </div>
                {selected.snapshot && (
                  <>
                    <div>
                      <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">Prediction Snapshot</div>
                      <div className="p-3 rounded-md bg-[#0F141C] border border-[#1E2633] flex items-center justify-between text-sm">
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="badge-dot" style={{ background: CLASSIFICATIONS[selected.snapshot.predicted_class]?.color }} />
                            <span className="font-semibold text-white">{CLASSIFICATIONS[selected.snapshot.predicted_class]?.label}</span>
                          </div>
                          <div className="text-[11px] text-slate-500 mt-1">Priority {selected.snapshot.review_priority?.toFixed(2)} · Confidence {(selected.snapshot.confidence*100).toFixed(1)}%</div>
                        </div>
                        <div className="text-[11px] mono text-slate-400">
                          W {(selected.snapshot.probs.review_warranted*100).toFixed(0)}% · NW {(selected.snapshot.probs.review_not_warranted*100).toFixed(0)}% · II {(selected.snapshot.probs.insufficient_information*100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">Evidence Summary</div>
                      <div className="grid grid-cols-5 gap-2 text-xs">
                        {Object.entries(selected.snapshot.evidence_summary || {}).map(([k, v]) => (
                          <div key={k} className="p-2 rounded-md bg-[#0F141C] border border-[#1E2633] text-center">
                            <div className="text-[10px] text-slate-500 uppercase tracking-wider">{k.replace("_", " ")}</div>
                            <div className="mono font-semibold text-white text-base">{v}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                )}
                {selected.notes && (
                  <div>
                    <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-2">Reviewer Notes</div>
                    <div className="p-3 rounded-md bg-[#0F141C] border border-[#1E2633] text-sm text-slate-300">{selected.notes}</div>
                  </div>
                )}
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
