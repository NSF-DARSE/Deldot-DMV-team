import React from "react";
import { useLocation } from "react-router-dom";
import { Search, Bell, User, Database, CircleDot } from "lucide-react";

const titleFor = (path) => {
  if (path === "/") return "Dashboard";
  if (path.startsWith("/queue")) return "Review Queue";
  if (path.startsWith("/warranted")) return "Review Warranted";
  if (path.startsWith("/not-warranted")) return "Review Not Warranted";
  if (path.startsWith("/insufficient")) return "Insufficient Information";
  if (path.startsWith("/analytics")) return "Analytics";
  if (path.startsWith("/model")) return "Model Performance";
  if (path.startsWith("/audit")) return "Audit History";
  if (path.startsWith("/case")) return "Case Investigation";
  if (path.startsWith("/import")) return "Import / Export";
  return "Hencheck";
};

export default function TopHeader({ phase, setPhase, search, setSearch }) {
  const loc = useLocation();
  return (
    <header
      data-testid="top-header"
      className="h-16 border-b border-[#1E2633] bg-[#0B0F16]/80 backdrop-blur-md sticky top-0 z-40 flex items-center px-6 gap-4"
    >
      <div className="flex-1">
        <div className="text-[10px] uppercase tracking-widest text-slate-500 font-medium">Delaware DMV · Hencheck</div>
        <h1 className="font-display text-lg font-semibold text-white leading-tight" data-testid="page-title">
          {titleFor(loc.pathname)}
        </h1>
      </div>

      <div className="relative">
        <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
        <input
          value={search || ""}
          onChange={(e) => setSearch?.(e.target.value)}
          placeholder="Search cases, audit, evidence…"
          data-testid="global-search"
          className="pl-9 pr-3 h-9 w-80 rounded-md bg-[#121821] border border-[#1E2633] text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-blue-500/50"
        />
      </div>

      <div className="flex items-center gap-1 bg-[#121821] border border-[#1E2633] rounded-md p-0.5" data-testid="phase-selector">
        {["T0", "T1"].map((ph) => (
          <button
            key={ph}
            onClick={() => setPhase?.(ph)}
            data-testid={`phase-${ph}`}
            className={`px-3 py-1.5 text-xs font-semibold rounded transition-colors ${
              phase === ph ? "bg-blue-600 text-white" : "text-slate-400 hover:text-white"
            }`}
          >
            {ph}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-2 px-3 h-9 rounded-md bg-[#121821] border border-[#1E2633] text-xs text-slate-300" data-testid="dataset-status">
        <Database className="w-3.5 h-3.5 text-blue-400" />
        <span>Dataset</span>
        <CircleDot className="w-3 h-3 text-emerald-400 animate-pulse" />
        <span className="text-emerald-400 font-medium">Live</span>
      </div>

      <button className="w-9 h-9 rounded-md bg-[#121821] border border-[#1E2633] flex items-center justify-center text-slate-400 hover:text-white transition-colors relative" data-testid="notifications">
        <Bell className="w-4 h-4" />
        <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full bg-red-500"></span>
      </button>
      <div className="flex items-center gap-2 pl-3 border-l border-[#1E2633]">
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white text-xs font-semibold">
          <User className="w-4 h-4" />
        </div>
        <div className="leading-tight hidden xl:block">
          <div className="text-xs text-slate-500">Analyst</div>
          <div className="text-xs mono font-semibold text-slate-200">analyst_demo</div>
        </div>
      </div>
    </header>
  );
}
