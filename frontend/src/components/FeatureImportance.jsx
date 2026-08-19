import React, { useEffect, useState } from "react";
import { fetchFeatureImportance, CLASSIFICATIONS } from "../lib/api";
import { TrendingUp, TrendingDown, Sparkles } from "lucide-react";

export default function FeatureImportance({ candidateId }) {
  const [data, setData] = useState(null);
  useEffect(() => { (async () => setData(await fetchFeatureImportance(candidateId)))(); }, [candidateId]);
  if (!data) return null;
  const cls = CLASSIFICATIONS[data.predicted_class];
  return (
    <div className="card-surface-elevated p-5" data-testid="feature-importance">
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold flex items-center gap-1.5">
            <Sparkles className="w-3 h-3 text-cyan-400" /> Model Reasoning
          </div>
          <h3 className="font-display text-base font-semibold text-white">Feature Importance</h3>
          <p className="text-xs text-slate-500 mt-1">
            Which of the 202 model features pushed this case toward{" "}
            <span className={cls?.text}>{cls?.label || data.predicted_class}</span>.
          </p>
        </div>
      </div>
      {(!data.features || data.features.length === 0) && (
        <div className="text-xs text-slate-500 text-center py-4">No feature contributions available.</div>
      )}
      <div className="space-y-2.5">
        {(data.features || []).slice(0, 8).map((f, i) => {
          const pushes = f.pushes_toward_class;
          const Icon = f.class_direction === "higher" ? TrendingUp : TrendingDown;
          const width = Math.min(100, Math.abs(f.case_z_score) * 25);
          return (
            <div key={i} className="p-2.5 rounded-md bg-[#0F141C] border border-[#1B222E]" data-testid={`feat-${f.feature}`}>
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2 min-w-0">
                  <div className={`w-6 h-6 rounded flex items-center justify-center shrink-0 ${
                    pushes ? "bg-red-500/15 border border-red-500/30" : "bg-emerald-500/15 border border-emerald-500/30"
                  }`}>
                    <Icon className={`w-3 h-3 ${pushes ? "text-red-400" : "text-emerald-400"}`} />
                  </div>
                  <span className="text-xs font-semibold text-white truncate mono">{f.label}</span>
                </div>
                <div className="text-[11px] mono text-slate-500 shrink-0 pl-2">
                  z={f.case_z_score >= 0 ? "+" : ""}{f.case_z_score.toFixed(2)}
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2 text-[10px] mb-1.5">
                <div><span className="text-slate-500">Case </span><span className="mono font-semibold text-white">{f.case_value}</span></div>
                <div><span className="text-slate-500">Class μ </span><span className="mono text-slate-300">{f.class_mean}</span></div>
                <div><span className="text-slate-500">All μ </span><span className="mono text-slate-500">{f.overall_mean}</span></div>
              </div>
              <div className="h-1 rounded-full bg-[#1B222E] overflow-hidden">
                <div className="h-full gauge-fill" style={{ width: `${width}%`, background: pushes ? "#EF4444" : "#10B981" }} />
              </div>
              <div className="text-[10px] text-slate-500 mt-1">
                Class typically has {f.class_direction} values ·{" "}
                <span className={pushes ? "text-red-400" : "text-emerald-400"}>
                  {pushes ? "pushes toward" : "against"} {cls?.short}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
