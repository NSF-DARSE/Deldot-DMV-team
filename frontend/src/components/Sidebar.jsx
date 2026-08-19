import React, { useState } from "react";
import { NavLink } from "react-router-dom";
import {
  LayoutDashboard, ListChecks, AlertOctagon, ShieldCheck, HelpCircle,
  BarChart3, Cpu, History, ChevronLeft, ChevronRight, Shield, Upload, GitCompare
} from "lucide-react";

const nav = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, exact: true, tid: "nav-dashboard" },
  { to: "/queue", label: "Review Queue", icon: ListChecks, tid: "nav-queue" },
  { to: "/warranted", label: "Review Warranted", icon: AlertOctagon, tid: "nav-warranted", accent: "#EF4444" },
  { to: "/not-warranted", label: "Review Not Warranted", icon: ShieldCheck, tid: "nav-not-warranted", accent: "#10B981" },
  { to: "/insufficient", label: "Insufficient Information", icon: HelpCircle, tid: "nav-insufficient", accent: "#F59E0B" },
  { to: "/compare", label: "Case Comparison", icon: GitCompare, tid: "nav-compare", accent: "#22D3EE" },
  { divider: true },
  { to: "/analytics", label: "Analytics", icon: BarChart3, tid: "nav-analytics" },
  { to: "/model", label: "Model Performance", icon: Cpu, tid: "nav-model" },
  { to: "/audit", label: "Audit History", icon: History, tid: "nav-audit" },
  { to: "/import-export", label: "Import / Export", icon: Upload, tid: "nav-import-export" },
];

export default function Sidebar({ collapsed, setCollapsed }) {
  return (
    <aside
      data-testid="sidebar"
      className={`shrink-0 border-r border-[#1E2633] bg-[#0B0F16] flex flex-col transition-all duration-200 ${
        collapsed ? "w-[68px]" : "w-[248px]"
      }`}
    >
      <div className="h-16 px-4 flex items-center border-b border-[#1E2633] gap-3">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500/30 to-blue-700/10 border border-blue-500/40 flex items-center justify-center shrink-0">
          <Shield className="w-4.5 h-4.5 text-blue-400" strokeWidth={2.2} />
        </div>
        {!collapsed && (
          <div className="leading-tight overflow-hidden">
            <div className="text-[13px] font-bold text-white tracking-tight">Hencheck</div>
            <div className="text-[10px] uppercase tracking-widest text-slate-500">DMV Review Ops</div>
          </div>
        )}
      </div>
      <nav className="flex-1 py-4 px-2 overflow-y-auto">
        {nav.map((item, i) => {
          if (item.divider) return <div key={`d-${i}`} className="my-3 mx-2 h-px bg-[#1E2633]" />;
          const Icon = item.icon;
          return (
            <NavLink key={item.to} to={item.to} end={item.exact} data-testid={item.tid}
              className={({ isActive }) =>
                `group flex items-center gap-3 px-3 py-2.5 rounded-md mb-1 text-sm font-medium transition-colors ${
                  isActive ? "bg-[#161D27] text-white border border-[#232C3B]" : "text-slate-400 hover:bg-[#121821] hover:text-white border border-transparent"
                }`}
              title={collapsed ? item.label : undefined}>
              <Icon className="w-4 h-4 shrink-0" style={item.accent ? { color: item.accent } : undefined} />
              {!collapsed && <span className="truncate">{item.label}</span>}
            </NavLink>
          );
        })}
      </nav>
      <button onClick={() => setCollapsed(!collapsed)} data-testid="sidebar-toggle"
        className="border-t border-[#1E2633] py-3 px-4 flex items-center justify-center gap-2 text-slate-500 hover:text-white hover:bg-[#121821] transition-colors">
        {collapsed ? <ChevronRight className="w-4 h-4" /> : (<><ChevronLeft className="w-4 h-4" /><span className="text-xs font-medium">Collapse</span></>)}
      </button>
    </aside>
  );
}
