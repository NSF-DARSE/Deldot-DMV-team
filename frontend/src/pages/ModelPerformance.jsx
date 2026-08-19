import React, { useEffect, useState } from "react";
import { fetchModelPerf } from "../lib/api";
import { CLASSIFICATIONS } from "../lib/api";
import { LineChart, Line, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine } from "recharts";
import { Cpu, AlertTriangle } from "lucide-react";

function Metric({ label, value, sub, accent }) {
  return (
    <div className="card-surface-elevated p-5">
      <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">{label}</div>
      <div className="stat-value text-3xl mt-1" style={{ color: accent || "#0F172A" }}>{value}</div>
      {sub && <div className="text-xs text-slate-500 mt-1">{sub}</div>}
    </div>
  );
}

export default function ModelPerformance() {
  const [p, setP] = useState(null);
  useEffect(() => { (async () => setP(await fetchModelPerf()))(); }, []);
  if (!p) return <div className="p-10 text-slate-500">Loading…</div>;

  const m = p.metrics;
  const labels = p.confusion_matrix.labels;
  const mat = p.confusion_matrix.matrix;
  const maxVal = Math.max(...mat.flat());

  return (
    <div className="space-y-5" data-testid="model-perf-page">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Governance</div>
          <h1 className="font-display text-2xl font-bold text-white flex items-center gap-2">
            <Cpu className="w-5 h-5 text-blue-400" /> Model Performance
          </h1>
        </div>
        <div className="text-xs mono text-slate-500">
          {p.version} · trained <span className="text-slate-300 font-semibold">{new Date(p.trained_at).toLocaleDateString()}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-4">
        <Metric label="Multiclass Log Loss" value={m.multiclass_log_loss.toFixed(3)} sub="Lower is better" />
        <Metric label="Brier Score" value={m.brier_score.toFixed(3)} sub="Lower is better" />
        <Metric label="Macro F1" value={m.macro_f1.toFixed(3)} sub="0 – 1" accent="#10B981" />
        <Metric label="Expected Calibration Err" value={m.expected_calibration_error.toFixed(3)} sub="Lower is better" accent="#F59E0B" />
        <Metric label="Accuracy" value={`${(m.accuracy*100).toFixed(1)}%`} sub="Holdout set" accent="#22D3EE" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Confusion matrix */}
        <div className="card-surface-elevated p-5">
          <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Diagnostic</div>
          <h3 className="font-display text-base font-semibold text-white mb-4">Confusion Matrix</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr>
                  <th className="p-2"></th>
                  {labels.map(l => <th key={l} className="p-2 text-slate-400 font-medium text-[11px]" style={{ color: CLASSIFICATIONS[l].color }}>Pred {CLASSIFICATIONS[l].short}</th>)}
                </tr>
              </thead>
              <tbody>
                {mat.map((row, i) => (
                  <tr key={i}>
                    <td className="p-2 text-slate-400 text-[11px] font-medium" style={{ color: CLASSIFICATIONS[labels[i]].color }}>Actual {CLASSIFICATIONS[labels[i]].short}</td>
                    {row.map((v, j) => {
                      const intensity = v / maxVal;
                      const bg = i === j ? `rgba(16,185,129,${0.15 + 0.5*intensity})` : `rgba(239,68,68,${0.1 + 0.4*intensity})`;
                      return (
                        <td key={j} className="p-3 text-center mono font-semibold text-white rounded" style={{ background: bg, border: "1px solid #1E2633" }}>{v}</td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Per-class */}
        <div className="card-surface-elevated p-5">
          <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Diagnostic</div>
          <h3 className="font-display text-base font-semibold text-white mb-4">Per-Class Performance</h3>
          <div className="space-y-4">
            {Object.entries(p.per_class).map(([k, v]) => {
              const meta = CLASSIFICATIONS[k];
              return (
                <div key={k}>
                  <div className="flex items-center justify-between mb-2 text-sm">
                    <div className="flex items-center gap-2">
                      <span className="badge-dot" style={{ background: meta.color }} />
                      <span className="font-semibold text-white">{meta.label}</span>
                    </div>
                    <span className="text-xs mono text-slate-500">n={v.support}</span>
                  </div>
                  <div className="grid grid-cols-3 gap-3">
                    {[["Precision", v.precision], ["Recall", v.recall], ["F1", v.f1]].map(([label, val]) => (
                      <div key={label}>
                        <div className="text-[10px] uppercase tracking-widest text-slate-500">{label}</div>
                        <div className="stat-value text-lg text-white">{val.toFixed(3)}</div>
                        <div className="h-1 rounded-full bg-[#1B222E] mt-1 overflow-hidden">
                          <div className="h-full gauge-fill" style={{ width: `${val*100}%`, background: meta.color }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Calibration */}
      <div className="card-surface-elevated p-5">
        <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Reliability</div>
        <h3 className="font-display text-base font-semibold text-white mb-4">Calibration Chart</h3>
        <div className="h-64"><ResponsiveContainer>
          <LineChart data={p.calibration} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1E2633" />
            <XAxis dataKey="bin" tick={{ fontSize: 11, fill: "#94A3B8" }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 11, fill: "#94A3B8" }} axisLine={false} tickLine={false} domain={[0, 1]} />
            <Tooltip contentStyle={{ background: "#121821", border: "1px solid #1E2633", borderRadius: 8 }} />
            <Legend wrapperStyle={{ fontSize: 11, color: "#CBD5E1" }} />
            <Line type="monotone" dataKey="expected" name="Perfectly Calibrated" stroke="#6B7280" strokeDasharray="4 4" dot={false} />
            <Line type="monotone" dataKey="observed" name="Observed" stroke="#3B82F6" strokeWidth={2} dot={{ fill: "#3B82F6", r: 4 }} />
          </LineChart>
        </ResponsiveContainer></div>
      </div>

      {/* Model comparison */}
      <div className="card-surface-elevated p-5">
        <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">History</div>
        <h3 className="font-display text-base font-semibold text-white mb-4">Model Comparison</h3>
        <div className="space-y-2">
          {p.model_comparison.map((m, i) => (
            <div key={i} className={`p-3 rounded-md flex items-center justify-between border ${m.current ? "border-blue-500/40 bg-blue-500/5" : "border-[#1E2633] bg-[#0F141C]"}`}>
              <div className="flex items-center gap-3">
                <div className={`w-8 h-8 rounded-md flex items-center justify-center ${m.current ? "bg-blue-500/20 text-blue-300" : "bg-[#1B222E] text-slate-500"}`}>
                  <Cpu className="w-4 h-4" />
                </div>
                <div>
                  <div className="text-sm font-semibold text-white">{m.name}</div>
                  {m.current && <div className="text-[10px] uppercase tracking-widest text-blue-400">Current Production</div>}
                </div>
              </div>
              <div className="flex items-center gap-6 mono text-xs">
                <div><span className="text-slate-500">F1 </span><span className="font-semibold text-white">{m.macro_f1.toFixed(3)}</span></div>
                <div><span className="text-slate-500">Acc </span><span className="font-semibold text-white">{(m.accuracy*100).toFixed(1)}%</span></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-lg border border-amber-800/30 bg-amber-950/20 p-4 flex items-start gap-3">
        <AlertTriangle className="w-4 h-4 text-amber-400 mt-0.5" />
        <div className="text-xs text-amber-200/80">
          <span className="font-semibold uppercase tracking-widest text-amber-300">Interpretation Note —</span> Low-confidence predictions should
          not be treated as certainty. Refer to per-class recall and calibration curves before over-relying on any single triage output.
        </div>
      </div>
    </div>
  );
}
