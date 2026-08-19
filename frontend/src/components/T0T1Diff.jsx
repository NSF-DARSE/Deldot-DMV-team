import React from "react";
import { ArrowRight, TrendingUp, TrendingDown, Minus, AlertCircle, CheckCircle2 } from "lucide-react";

const FIELDS = [
  { key: "days_observed", label: "Days Observed", type: "int", direction: "neutral" },
  { key: "out_of_state_tag_days", label: "Out-of-State Tag Days", type: "int", direction: "escalate_up" },
  { key: "primary_state_detected", label: "Primary State Detected", type: "str", direction: "categorical" },
  { key: "toll_gantry_hits", label: "Toll Gantry Hits", type: "int", direction: "escalate_up" },
  { key: "dl_state_match", label: "DL State Match", type: "bool", direction: "escalate_when_false" },
  { key: "neighborhood_parking_frequency", label: "Neighborhood Parking Freq.", type: "float", direction: "escalate_up" },
  { key: "vehicle_registration_status", label: "Vehicle Registration Status", type: "str", direction: "categorical" },
];

function fmt(v, type) {
  if (type === "bool") return v ? "Yes" : "No";
  if (type === "float") return typeof v === "number" ? v.toFixed(2) : v;
  return String(v);
}

function classify(field, a, b) {
  if (a === b) return { kind: "unchanged" };
  if (field.type === "bool") {
    if (field.direction === "escalate_when_false") {
      if (a === true && b === false) return { kind: "escalate" };
      if (a === false && b === true) return { kind: "resolve" };
    }
    return { kind: "change" };
  }
  if (field.type === "str" || field.direction === "categorical") {
    return { kind: "change" };
  }
  // numeric
  const delta = b - a;
  if (field.direction === "escalate_up") {
    return { kind: delta > 0 ? "escalate" : "resolve", delta };
  }
  return { kind: delta === 0 ? "unchanged" : delta > 0 ? "up" : "down", delta };
}

function KindIcon({ kind }) {
  if (kind === "escalate") return <TrendingUp className="w-4 h-4 text-red-600" />;
  if (kind === "resolve") return <TrendingDown className="w-4 h-4 text-emerald-600" />;
  if (kind === "up") return <TrendingUp className="w-4 h-4 text-blue-600" />;
  if (kind === "down") return <TrendingDown className="w-4 h-4 text-blue-600" />;
  if (kind === "change") return <AlertCircle className="w-4 h-4 text-amber-600" />;
  return <Minus className="w-4 h-4 text-slate-400" />;
}

function badgeFor(kind) {
  if (kind === "escalate") return "bg-red-50 text-red-700 border-red-200";
  if (kind === "resolve") return "bg-emerald-50 text-emerald-700 border-emerald-200";
  if (kind === "change") return "bg-amber-50 text-amber-700 border-amber-200";
  if (kind === "up" || kind === "down") return "bg-blue-50 text-blue-700 border-blue-200";
  return "bg-slate-50 text-slate-600 border-slate-200";
}

const kindLabel = {
  escalate: "Escalating",
  resolve: "Resolving",
  change: "Changed",
  up: "Increased",
  down: "Decreased",
  unchanged: "Unchanged",
};

export default function T0T1Diff({ t0 = {}, t1 = {} }) {
  const escCount = FIELDS.reduce((n, f) => n + (classify(f, t0[f.key], t1[f.key]).kind === "escalate" ? 1 : 0), 0);
  const resCount = FIELDS.reduce((n, f) => n + (classify(f, t0[f.key], t1[f.key]).kind === "resolve" ? 1 : 0), 0);
  const chgCount = FIELDS.reduce((n, f) => {
    const k = classify(f, t0[f.key], t1[f.key]).kind;
    return n + (k === "change" ? 1 : 0);
  }, 0);

  return (
    <div className="space-y-4" data-testid="t0-t1-diff">
      {/* Summary row */}
      <div className="grid grid-cols-3 gap-3">
        <div className="p-3 rounded-lg border border-red-200 bg-red-50">
          <div className="text-[11px] uppercase tracking-wider text-red-700 font-medium">Escalating</div>
          <div className="stat-value text-2xl text-red-800">{escCount}</div>
        </div>
        <div className="p-3 rounded-lg border border-emerald-200 bg-emerald-50">
          <div className="text-[11px] uppercase tracking-wider text-emerald-700 font-medium">Resolving</div>
          <div className="stat-value text-2xl text-emerald-800">{resCount}</div>
        </div>
        <div className="p-3 rounded-lg border border-amber-200 bg-amber-50">
          <div className="text-[11px] uppercase tracking-wider text-amber-700 font-medium">Changed</div>
          <div className="stat-value text-2xl text-amber-800">{chgCount}</div>
        </div>
      </div>

      <div className="rounded-lg border border-slate-200 overflow-hidden bg-white">
        <div className="grid grid-cols-12 bg-slate-50 border-b border-slate-200 px-4 py-2 text-[11px] uppercase tracking-wider font-semibold text-slate-500">
          <div className="col-span-3">Evidence Field</div>
          <div className="col-span-3">T0 (Initial Observation)</div>
          <div className="col-span-1 text-center"></div>
          <div className="col-span-3">T1 (Subsequent Observation)</div>
          <div className="col-span-2 text-right">Change</div>
        </div>
        {FIELDS.map((f) => {
          const a = t0[f.key];
          const b = t1[f.key];
          const c = classify(f, a, b);
          return (
            <div
              key={f.key}
              className="grid grid-cols-12 px-4 py-3 border-b border-slate-100 last:border-b-0 items-center row-hover"
              data-testid={`diff-row-${f.key}`}
            >
              <div className="col-span-3 text-sm text-slate-700">{f.label}</div>
              <div className="col-span-3 mono text-sm text-slate-800">{fmt(a, f.type)}</div>
              <div className="col-span-1 flex items-center justify-center">
                <ArrowRight className="w-4 h-4 text-slate-400" />
              </div>
              <div className="col-span-3 mono text-sm text-slate-900 font-semibold">{fmt(b, f.type)}</div>
              <div className="col-span-2 flex items-center justify-end gap-2">
                <KindIcon kind={c.kind} />
                <span
                  className={`text-[11px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded border ${badgeFor(
                    c.kind
                  )}`}
                >
                  {kindLabel[c.kind]}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
