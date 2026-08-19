import React from "react";
import { priorityBand } from "../lib/api";

export default function PriorityGauge({ value = 0, size = 180 }) {
  const clamped = Math.max(0, Math.min(1, value));
  const band = priorityBand(clamped);
  const r = size / 2 - 14;
  const cx = size / 2; const cy = size / 2;
  const circ = Math.PI * r;
  const dashOffset = circ * (1 - clamped);
  return (
    <div className="flex flex-col items-center" data-testid="priority-gauge">
      <svg width={size} height={size / 2 + 20} viewBox={`0 0 ${size} ${size / 2 + 20}`}>
        <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill="none" stroke="#1E2633" strokeWidth="14" strokeLinecap="round" />
        <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill="none" stroke={band.color} strokeWidth="14" strokeLinecap="round"
          strokeDasharray={circ} strokeDashoffset={dashOffset}
          style={{ transition: "stroke-dashoffset 900ms cubic-bezier(0.16, 1, 0.3, 1)", filter: `drop-shadow(0 0 8px ${band.color}66)` }} />
        <text x={cx} y={cy - 8} textAnchor="middle" style={{ fontSize: 32, fontWeight: 800, fill: "#F3F4F6", letterSpacing: "-0.03em" }}>
          {clamped.toFixed(2)}
        </text>
        <text x={cx} y={cy + 10} textAnchor="middle" style={{ fontSize: 10, fill: "#6B7280", letterSpacing: "0.12em", textTransform: "uppercase", fontWeight: 600 }}>
          Priority Index
        </text>
      </svg>
      <div className={`mt-1 px-3 py-1 rounded-full text-xs font-semibold border ${band.bg} ${band.text} ${band.border}`} data-testid="priority-band-label">
        {band.label} Priority
      </div>
    </div>
  );
}
