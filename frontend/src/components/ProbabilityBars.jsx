import React from "react";
import { CLASSIFICATIONS } from "../lib/api";
const ORDER = ["review_warranted", "review_not_warranted", "insufficient_information"];
export default function ProbabilityBars({ probs, predicted }) {
  const values = {
    review_warranted: probs.prob_review_warranted ?? 0,
    review_not_warranted: probs.prob_review_not_warranted ?? 0,
    insufficient_information: probs.prob_insufficient_information ?? 0,
  };
  return (
    <div className="space-y-3" data-testid="probability-bars">
      {ORDER.map((k) => {
        const v = values[k]; const meta = CLASSIFICATIONS[k]; const isPred = predicted === k;
        return (
          <div key={k} data-testid={`prob-row-${k}`}>
            <div className="flex items-center justify-between mb-1.5">
              <div className="flex items-center gap-2">
                <span className="badge-dot" style={{ background: meta.color }} />
                <span className={`text-sm ${isPred ? "font-semibold text-white" : "font-medium text-slate-300"}`}>{meta.label}</span>
                {isPred && <span className="text-[9px] uppercase tracking-widest font-semibold text-slate-400 px-1.5 py-0.5 border border-[#232C3B] rounded">Predicted</span>}
              </div>
              <span className="mono text-sm font-semibold text-white">{(v * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2 rounded-full bg-[#1B222E] overflow-hidden">
              <div className="h-full gauge-fill" style={{ width: `${v * 100}%`, background: meta.color, boxShadow: `0 0 8px ${meta.color}55` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
