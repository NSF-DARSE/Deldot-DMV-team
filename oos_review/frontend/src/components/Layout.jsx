import React, { useState } from "react";
import Sidebar from "./Sidebar";
import TopHeader from "./TopHeader";
import { AlertTriangle } from "lucide-react";

export default function Layout({ children }) {
  const [collapsed, setCollapsed] = useState(false);
  const [phase, setPhase] = useState("T1");
  const [search, setSearch] = useState("");

  return (
    <div className="min-h-screen bg-[#F5F7FA] text-slate-800 flex" data-testid="app-layout">
      <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} />
      <div className="flex-1 flex flex-col min-w-0">
        <TopHeader phase={phase} setPhase={setPhase} search={search} setSearch={setSearch} />
        <div className="px-4 py-2 bg-amber-950/20 border-b border-amber-800/30 flex items-start gap-2 text-[11px] text-amber-200/80" data-testid="disclaimer-banner">
          <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0 text-amber-500" />
          <span>
            <span className="font-semibold uppercase tracking-widest text-amber-300">Decision Support Only —</span>{" "}
            This application supports human reviewers. It does not automatically determine residency, violations, penalties, fees, guilt, or enforcement actions.
          </span>
        </div>
        <main className="flex-1 px-6 py-6 overflow-x-hidden">
          {typeof children === "function" ? children({ phase, search }) : children}
        </main>
      </div>
    </div>
  );
}
