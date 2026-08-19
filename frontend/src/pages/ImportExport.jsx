import React from "react";
import { Download, Database, FileText, AlertTriangle } from "lucide-react";
import { exportCsvUrl } from "../lib/api";

export default function ImportExport() {
  return (
    <div className="space-y-5" data-testid="import-export-page">
      <div>
        <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Data Operations</div>
        <h1 className="font-display text-2xl font-bold text-white">Export & Data Sources</h1>
        <p className="text-sm text-slate-400 mt-1">
          Predictions are loaded directly from the repository&apos;s <code className="mono text-slate-300">outputs/case_predictions.csv</code>{" "}
          and enriched with the evidence timeline. Analyst decisions and tags round-trip back through export.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="card-surface-elevated p-5">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-md bg-blue-500/10 border border-blue-500/30 flex items-center justify-center">
              <Database className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Source</div>
              <h3 className="font-display text-base font-semibold text-white">Live Repository Data</h3>
            </div>
          </div>
          <ul className="space-y-2 text-sm text-slate-300">
            <li className="flex items-start gap-2"><FileText className="w-3.5 h-3.5 mt-1 text-blue-400" /> <span><code className="mono">outputs/case_predictions.csv</code> — 24,000 prediction rows (12,000 candidates × T0 / T1)</span></li>
            <li className="flex items-start gap-2"><FileText className="w-3.5 h-3.5 mt-1 text-blue-400" /> <span><code className="mono">data/processed/evidence_timeline.csv.gz</code> — 216,000 evidence records across 5 sources</span></li>
            <li className="flex items-start gap-2"><FileText className="w-3.5 h-3.5 mt-1 text-blue-400" /> <span><code className="mono">models/metrics.json</code> — 5-fold OOF metrics for the selected model</span></li>
            <li className="flex items-start gap-2"><FileText className="w-3.5 h-3.5 mt-1 text-blue-400" /> <span><code className="mono">outputs/prediction_metadata.json</code> — model version + timestamp</span></li>
          </ul>
          <div className="mt-4 rounded-md bg-blue-950/30 border border-blue-800/30 p-3 flex items-start gap-2 text-xs text-blue-200/80">
            <AlertTriangle className="w-3.5 h-3.5 text-blue-400 mt-0.5 shrink-0" />
            <span>Data is validated at startup — check the <strong>Data Status</strong> panel on the Dashboard.</span>
          </div>
        </div>

        <div className="card-surface-elevated p-5">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-md bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
              <Download className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Dashboard → Submission</div>
              <h3 className="font-display text-base font-semibold text-white">Export case_predictions.csv</h3>
            </div>
          </div>
          <p className="text-sm text-slate-400 leading-relaxed mb-4">
            Download the full 24,000-row dataset in the original submission schema:
            <code className="mono text-slate-300 block mt-2 text-xs">candidate_record_id, phase, predicted_class, p_review_warranted, p_review_not_warranted, p_insufficient_evidence, review_priority</code>
          </p>
          <a href={exportCsvUrl()}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-md bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-semibold transition-colors"
            data-testid="export-download">
            <Download className="w-4 h-4" /> Download CSV
          </a>
          <div className="mt-6 rounded-md bg-amber-950/30 border border-amber-800/30 p-3 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" />
            <div className="text-xs text-amber-200/80">
              These outputs are triage aids — not automated legal, residency, tax, fee, or enforcement determinations.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
