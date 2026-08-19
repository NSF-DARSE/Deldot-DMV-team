import React, { useEffect, useState } from "react";
import { fetchStats, CLASSIFICATIONS } from "../lib/api";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from "recharts";

export default function Analytics() {
  const [s, setS] = useState(null);
  useEffect(() => { (async () => setS(await fetchStats()))(); }, []);
  if (!s) return <div className="p-10 text-slate-500">Loading…</div>;

  const donut = Object.entries(s.by_class).map(([k, v]) => ({ name: CLASSIFICATIONS[k].label, value: v, color: CLASSIFICATIONS[k].color }));
  const priorityBins = s.priority_bins.map((n, i) => ({ bin: `${(i/10).toFixed(1)}`, count: n }));
  const evStrength = Object.entries(s.evidence_strength).map(([k, v]) => ({ name: k, count: v, color: k === "Strong" ? "#10B981" : k === "Moderate" ? "#F59E0B" : "#6B7280" }));
  const evCoverage = Object.entries(s.evidence_coverage).map(([k, v]) => ({ subject: k.replace("_", " "), A: v, fullMark: s.total }));
  const changes = [
    { name: "W→NW", count: s.changes.warranted_to_not, color: "#10B981" },
    { name: "NW→W", count: s.changes.not_to_warranted, color: "#EF4444" },
    { name: "II→W", count: s.changes.ii_to_warranted, color: "#F97316" },
    { name: "II→NW", count: s.changes.ii_to_not, color: "#22D3EE" },
    { name: "→II", count: s.changes.to_ii, color: "#F59E0B" },
    { name: "None", count: s.changes.no_change, color: "#6B7280" },
  ];

  return (
    <div className="space-y-5" data-testid="analytics-page">
      <div>
        <div className="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Executive</div>
        <h1 className="font-display text-2xl font-bold text-white">Analytics Dashboard</h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="card-surface-elevated p-5">
          <h3 className="font-display text-base font-semibold text-white mb-3">Case Distribution</h3>
          <div className="h-56"><ResponsiveContainer>
            <PieChart>
              <Pie data={donut} innerRadius={55} outerRadius={80} paddingAngle={2} dataKey="value">
                {donut.map((d, i) => <Cell key={i} fill={d.color} stroke="#0B0F16" strokeWidth={2} />)}
              </Pie>
              <Tooltip contentStyle={{ background: "#121821", border: "1px solid #1E2633", borderRadius: 8 }} />
            </PieChart>
          </ResponsiveContainer></div>
        </div>

        <div className="card-surface-elevated p-5 lg:col-span-2">
          <h3 className="font-display text-base font-semibold text-white mb-3">Review Priority Distribution</h3>
          <div className="h-56"><ResponsiveContainer>
            <BarChart data={priorityBins} margin={{ top: 10, right: 10, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E2633" vertical={false} />
              <XAxis dataKey="bin" tick={{ fontSize: 11, fill: "#94A3B8" }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fill: "#94A3B8" }} axisLine={false} tickLine={false} />
              <Tooltip cursor={{ fill: "#161D27" }} contentStyle={{ background: "#121821", border: "1px solid #1E2633", borderRadius: 8 }} />
              <Bar dataKey="count" radius={[4,4,0,0]}>
                {priorityBins.map((_, i) => <Cell key={i} fill={i >= 8 ? "#EF4444" : i >= 7 ? "#F97316" : i >= 4 ? "#F59E0B" : "#10B981"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer></div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="card-surface-elevated p-5">
          <h3 className="font-display text-base font-semibold text-white mb-3">T0 → T1 Classification Changes</h3>
          <div className="h-56"><ResponsiveContainer>
            <BarChart data={changes} margin={{ top: 10, right: 10, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E2633" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#94A3B8" }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fill: "#94A3B8" }} axisLine={false} tickLine={false} />
              <Tooltip cursor={{ fill: "#161D27" }} contentStyle={{ background: "#121821", border: "1px solid #1E2633", borderRadius: 8 }} />
              <Bar dataKey="count" radius={[4,4,0,0]}>
                {changes.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer></div>
        </div>

        <div className="card-surface-elevated p-5">
          <h3 className="font-display text-base font-semibold text-white mb-3">Cases by Evidence Strength</h3>
          <div className="h-56"><ResponsiveContainer>
            <BarChart data={evStrength} layout="vertical" margin={{ top: 10, right: 20, bottom: 0, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E2633" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11, fill: "#94A3B8" }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: "#94A3B8" }} axisLine={false} tickLine={false} />
              <Tooltip cursor={{ fill: "#161D27" }} contentStyle={{ background: "#121821", border: "1px solid #1E2633", borderRadius: 8 }} />
              <Bar dataKey="count" radius={[0,4,4,0]}>
                {evStrength.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer></div>
        </div>
      </div>

      <div className="card-surface-elevated p-5">
        <h3 className="font-display text-base font-semibold text-white mb-3">Evidence Source Coverage</h3>
        <div className="h-72"><ResponsiveContainer>
          <RadarChart data={evCoverage}>
            <PolarGrid stroke="#232C3B" />
            <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: "#CBD5E1" }} />
            <PolarRadiusAxis stroke="#1E2633" tick={{ fontSize: 10, fill: "#64748B" }} />
            <Radar name="Coverage" dataKey="A" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.35} />
            <Tooltip contentStyle={{ background: "#121821", border: "1px solid #1E2633", borderRadius: 8 }} />
          </RadarChart>
        </ResponsiveContainer></div>
      </div>
    </div>
  );
}
