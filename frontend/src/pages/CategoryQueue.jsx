import React, { useEffect, useState } from "react";
import Queue from "./Queue";
import { fetchCases } from "../lib/api";

export default function CategoryQueue({ classKey, title, accent }) {
  const [summary, setSummary] = useState(null);
  useEffect(() => {
    (async () => {
      const d = await fetchCases({ predicted_class: classKey, limit: 5000 });
      const cases = d.cases;
      const avgPriority = cases.length ? cases.reduce((s, c) => s + c.review_priority, 0) / cases.length : 0;
      const avgConf = cases.length ? cases.reduce((s, c) => s + c.confidence, 0) / cases.length : 0;
      const top = [...cases].sort((a, b) => b.review_priority - a.review_priority).slice(0, 5);
      setSummary({ total: cases.length, avgPriority, avgConf, top });
    })();
  }, [classKey]);

  return (
    <div className="space-y-5">
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="card-surface-elevated p-5">
            <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Total</div>
            <div className="stat-value text-3xl mt-1" style={{ color: accent }}>{summary.total}</div>
            <div className="text-xs text-slate-500 mt-1">cases in category</div>
          </div>
          <div className="card-surface-elevated p-5">
            <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Avg Priority</div>
            <div className="stat-value text-3xl mt-1 text-white">{summary.avgPriority.toFixed(2)}</div>
            <div className="text-xs text-slate-500 mt-1">0.00 – 1.00</div>
          </div>
          <div className="card-surface-elevated p-5">
            <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Avg Confidence</div>
            <div className="stat-value text-3xl mt-1 text-white">{(summary.avgConf*100).toFixed(1)}%</div>
            <div className="text-xs text-slate-500 mt-1">model probability</div>
          </div>
          <div className="card-surface-elevated p-5">
            <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Top Priority</div>
            <div className="space-y-1 mt-2">
              {summary.top.slice(0,3).map(c => (
                <div key={c.candidate_id} className="flex items-center justify-between text-xs">
                  <span className="mono text-slate-300 font-semibold">{c.candidate_id}</span>
                  <span className="mono font-semibold" style={{ color: accent }}>{c.review_priority.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      <Queue presetClass={classKey} title={title} accent={accent} />
    </div>
  );
}
