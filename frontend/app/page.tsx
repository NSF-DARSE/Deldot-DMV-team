"use client";

import { useEffect, useMemo, useState } from "react";

type ReviewClass = "review_warranted" | "review_not_warranted" | "insufficient_evidence";
type Phase = "T0" | "T1";
type CaseSummary = {
  candidate_record_id: string; phase: Phase; predicted_class: ReviewClass;
  p_review_warranted: number; p_review_not_warranted: number;
  p_insufficient_evidence: number; review_priority: number;
  evidence_strength: number; confidence: number;
};
type PhaseDetail = CaseSummary & {
  evidence_recency: number; cross_source_agreement: number; conflict_score: number;
  dominant_evidence_state: string; independent_source_types: number; new_t1_records: number;
  explanation: { summary: string; evidence_bullets: string[] };
};
type CaseDetail = {
  candidate_record_id: string; t0: PhaseDetail; t1: PhaseDetail;
  changes: { predicted_class_changed: boolean; p_review_warranted_change: number; priority_change: number; evidence_strength_change: number; new_evidence_records: number };
};
type TimelineEvent = { phase_available: Phase; effective_source_domain: string; source_record_id: string; event_date: string | null; observed_date: string | null; state: string; event_type: string; status: string; quality: string; record_action: string; identity_consistency: number };
type Explanation = { summary: string; evidence_bullets: string[]; feature_contributions: { feature: string; contribution: number; direction: string }[] };

const fallbackCases: CaseSummary[] = [
  { candidate_record_id: "CAN-GE61791MG4", phase: "T1", predicted_class: "review_warranted", p_review_warranted: .5872, p_review_not_warranted: .137, p_insufficient_evidence: .2758, review_priority: .559, evidence_strength: .9773, confidence: .5872 },
  { candidate_record_id: "CAN-BL7OZ9KMQ6", phase: "T1", predicted_class: "review_warranted", p_review_warranted: .5879, p_review_not_warranted: .1629, p_insufficient_evidence: .2493, review_priority: .5573, evidence_strength: .9745, confidence: .5879 },
  { candidate_record_id: "CAN-YKDMTNSC7N", phase: "T1", predicted_class: "review_warranted", p_review_warranted: .5813, p_review_not_warranted: .1385, p_insufficient_evidence: .2802, review_priority: .5539, evidence_strength: .9773, confidence: .5813 },
  { candidate_record_id: "CAN-CNF2U0FWZ8", phase: "T1", predicted_class: "review_warranted", p_review_warranted: .5794, p_review_not_warranted: .1271, p_insufficient_evidence: .2935, review_priority: .5499, evidence_strength: .9584, confidence: .5794 },
];
const labels: Record<ReviewClass, string> = { review_warranted: "Review warranted", review_not_warranted: "Review not warranted", insufficient_evidence: "Insufficient evidence" };
const percent = (value: number) => `${Math.round(value * 100)}%`;
const signed = (value: number) => `${value >= 0 ? "+" : ""}${Math.round(value * 100)} pts`;

export default function Home() {
  const [phase, setPhase] = useState<Phase>("T1");
  const [cases, setCases] = useState<CaseSummary[]>(fallbackCases);
  const [connection, setConnection] = useState<"connecting" | "live" | "preview">("connecting");
  const [classFilter, setClassFilter] = useState<ReviewClass | "all">("all");
  const [priorityFilter, setPriorityFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState(fallbackCases[0].candidate_record_id);
  const [detail, setDetail] = useState<CaseDetail | null>(null);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [actionMessage, setActionMessage] = useState("");
  const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

  useEffect(() => {
    const controller = new AbortController();
    setConnection("connecting");
    fetch(`${apiBase}/priority?phase=${phase}&limit=100`, { signal: controller.signal })
      .then((response) => { if (!response.ok) throw new Error("API unavailable"); return response.json(); })
      .then((payload) => { setCases(payload.items); setSelectedId(payload.items[0]?.candidate_record_id ?? ""); setConnection("live"); })
      .catch(() => { setCases(fallbackCases.map((item) => ({ ...item, phase }))); setConnection("preview"); });
    return () => controller.abort();
  }, [apiBase, phase]);

  useEffect(() => {
    if (!selectedId || connection !== "live") return;
    const controller = new AbortController();
    Promise.all([
      fetch(`${apiBase}/cases/${selectedId}`, { signal: controller.signal }).then((response) => response.json()),
      fetch(`${apiBase}/cases/${selectedId}/timeline`, { signal: controller.signal }).then((response) => response.json()),
      fetch(`${apiBase}/cases/${selectedId}/explanation?phase=${phase}`, { signal: controller.signal }).then((response) => response.json()),
    ]).then(([casePayload, timelinePayload, explanationPayload]) => {
      setDetail(casePayload); setTimeline(timelinePayload.events); setExplanation(explanationPayload); setActionMessage("");
    }).catch(() => undefined);
    return () => controller.abort();
  }, [apiBase, selectedId, phase, connection]);

  const visibleCases = useMemo(() => cases.filter((item) => {
    const classMatches = classFilter === "all" || item.predicted_class === classFilter;
    const queryMatches = item.candidate_record_id.toLowerCase().includes(query.toLowerCase());
    const priorityMatches = priorityFilter === "all" || (priorityFilter === "high" && item.review_priority >= .45) || (priorityFilter === "medium" && item.review_priority >= .30 && item.review_priority < .45) || (priorityFilter === "low" && item.review_priority < .30);
    return classMatches && queryMatches && priorityMatches;
  }), [cases, classFilter, priorityFilter, query]);
  const selected = cases.find((item) => item.candidate_record_id === selectedId) ?? visibleCases[0];
  const selectedPhase = detail?.[phase.toLowerCase() as "t0" | "t1"];
  const bullets = explanation?.evidence_bullets ?? selectedPhase?.explanation.evidence_bullets ?? ["Multiple source types contribute to this recommendation.", "Probability uncertainty remains visible to the reviewer.", "T0 and T1 evidence snapshots are preserved separately."];

  const recordAction = (action: string) => {
    if (connection !== "live") { setActionMessage("Preview mode: connect the FastAPI service to record workflow actions."); return; }
    fetch(`${apiBase}/cases/${selectedId}/review-action`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ phase, action }) })
      .then((response) => response.json()).then((payload) => setActionMessage(payload.message)).catch(() => setActionMessage("The workflow action could not be recorded."));
  };

  return <main className="app-shell">
    <aside className="sidebar">
      <div className="agency-mark"><span>DE</span><div><strong>Delaware DMV</strong><small>Review support</small></div></div>
      <nav aria-label="Primary navigation"><button className="nav-item active"><span>01</span> Review queue</button><button className="nav-item"><span>02</span> Model performance</button><button className="nav-item"><span>03</span> Audit history</button></nav>
      <div className="sidebar-note"><span className="shield">i</span><p><strong>Human decision required</strong>This tool organizes evidence. It never initiates enforcement.</p></div>
    </aside>

    <section className="workspace">
      <header className="topbar"><div><p className="eyebrow">Potential out-of-state tag holder review</p><h1>Review Queue</h1></div><div className={`connection ${connection}`}><span />{connection === "live" ? "Live case data" : connection === "preview" ? "Preview data" : "Connecting"}</div></header>
      <div className="guardrail"><strong>Decision support only</strong><span>Recommendations prioritize staff review and are not legal, residency, fee, or enforcement determinations.</span></div>
      <div className="summary-grid"><article><small>Queue phase</small><strong>{phase}</strong><span>{phase === "T1" ? "Later evidence included" : "Initial evidence only"}</span></article><article><small>Cases loaded</small><strong>{connection === "live" ? "100" : visibleCases.length}</strong><span>Ordered by review priority</span></article><article><small>Top priority</small><strong>{percent(cases[0]?.review_priority ?? 0)}</strong><span>Operational queue score</span></article></div>

      <div className="content-grid">
        <section className="queue-card">
          <div className="queue-toolbar"><div className="phase-switch" aria-label="Evidence phase">{(["T0", "T1"] as const).map((value) => <button key={value} className={phase === value ? "active" : ""} onClick={() => setPhase(value)}>{value}</button>)}</div><input aria-label="Search candidate ID" placeholder="Search candidate ID" value={query} onChange={(event) => setQuery(event.target.value)} /><select aria-label="Filter by priority" value={priorityFilter} onChange={(event) => setPriorityFilter(event.target.value)}><option value="all">All priorities</option><option value="high">High · 0.45+</option><option value="medium">Medium · 0.30–0.44</option><option value="low">Low · under 0.30</option></select><select aria-label="Filter by recommendation" value={classFilter} onChange={(event) => setClassFilter(event.target.value as ReviewClass | "all")}><option value="all">All recommendations</option><option value="review_warranted">Review warranted</option><option value="review_not_warranted">Review not warranted</option><option value="insufficient_evidence">Insufficient evidence</option></select></div>
          <div className="table-head"><span>Candidate</span><span>Recommendation</span><span>Confidence</span><span>Priority</span></div>
          <div className="case-list">{visibleCases.length ? visibleCases.map((item) => <button key={item.candidate_record_id} className={`case-row ${selectedId === item.candidate_record_id ? "selected" : ""}`} onClick={() => setSelectedId(item.candidate_record_id)}><span className="candidate"><i>{item.candidate_record_id.slice(-2)}</i><span><strong>{item.candidate_record_id}</strong><small>{item.phase} · evidence {percent(item.evidence_strength)}</small></span></span><span><em className={`status ${item.predicted_class}`}>{labels[item.predicted_class]}</em></span><span className="metric"><strong>{percent(item.confidence)}</strong><i><b style={{ width: percent(item.confidence) }} /></i></span><span className="priority-score">{item.review_priority.toFixed(2)}</span></button>) : <div className="empty-state">No cases match the current filters.</div>}</div>
        </section>

        <aside className="detail-card"><div className="detail-kicker"><span>Case detail</span><em>{selected?.phase ?? phase}</em></div><h2>{selected?.candidate_record_id ?? "Select a case"}</h2>{selected && <><section className="recommendation"><small>System recommendation</small><h3>{labels[selected.predicted_class]}</h3><div><span><strong>{percent(selected.confidence)}</strong> confidence</span><span><strong>{selected.review_priority.toFixed(2)}</strong> priority</span></div></section><section className="evidence-preview"><div className="section-title"><h3>Evidence summary</h3><span>{percent(selected.evidence_strength)} strength</span></div><ul>{bullets.slice(0, 4).map((bullet) => <li key={bullet}><i>●</i>{bullet}</li>)}</ul></section><button className="open-case" onClick={() => setExpanded(true)}>Open complete case review <span>→</span></button></>}</aside>
      </div>
    </section>

    {expanded && selected && <div className="drawer-backdrop" role="presentation" onMouseDown={() => setExpanded(false)}><section className="case-drawer" role="dialog" aria-modal="true" aria-labelledby="case-title" onMouseDown={(event) => event.stopPropagation()}>
      <header className="drawer-header"><div><p className="eyebrow">Human review workspace · {phase}</p><h2 id="case-title">{selected.candidate_record_id}</h2></div><button aria-label="Close case detail" onClick={() => setExpanded(false)}>×</button></header>
      <div className="drawer-guardrail"><strong>No automated enforcement</strong><span>Record a workflow disposition only after reviewing the underlying evidence.</span></div>
      <div className="drawer-body">
        <section className="drawer-main">
          <article className="decision-card"><div><small>Recommendation</small><h3>{labels[selected.predicted_class]}</h3><p>{explanation?.summary ?? selectedPhase?.explanation.summary ?? "The full deterministic explanation is available when the API is connected."}</p></div><div className="decision-numbers"><span><strong>{percent(selected.confidence)}</strong>Confidence</span><span><strong>{selected.review_priority.toFixed(2)}</strong>Priority</span></div></article>
          <article className="panel"><div className="panel-heading"><h3>Evidence summary</h3><span>{selectedPhase?.independent_source_types ?? 6} source types</span></div><ul className="evidence-list">{bullets.map((bullet) => <li key={bullet}><span>✓</span>{bullet}</li>)}</ul></article>
          <article className="panel"><div className="panel-heading"><h3>Evidence timeline</h3><span>Chronological · identity fields suppressed</span></div><div className="timeline">{timeline.filter((event) => phase === "T1" || event.phase_available === "T0").slice(-12).reverse().map((event) => <div className="timeline-event" key={`${event.source_record_id}-${event.event_date}`}><span className={`timeline-dot ${event.phase_available.toLowerCase()}`} /><time>{event.event_date ?? "Date unavailable"}</time><div><strong>{event.effective_source_domain}</strong><p>{event.event_type || event.record_action || "Evidence record"}{event.state ? ` · ${event.state}` : " · state unavailable"}</p></div><em>{event.phase_available}</em></div>)}</div></article>
          {explanation?.feature_contributions?.length ? <article className="panel"><div className="panel-heading"><h3>Model associations</h3><span>Coefficient contributions · non-causal</span></div><div className="contributions">{explanation.feature_contributions.slice(0, 6).map((item) => <div key={item.feature}><span>{item.feature.replaceAll("_", " ")}</span><i className={item.contribution >= 0 ? "positive" : "negative"}><b style={{ width: `${Math.min(100, Math.abs(item.contribution) * 22)}%` }} /></i><em>{item.direction}</em></div>)}</div></article> : null}
        </section>
        <aside className="drawer-side">
          <article className="panel comparison"><div className="panel-heading"><h3>T0 → T1</h3><span>{detail?.changes.predicted_class_changed ? "Class changed" : "Class stable"}</span></div>{detail ? <><div className="phase-card"><small>T0 · initial evidence</small><strong>{labels[detail.t0.predicted_class]}</strong><span>{percent(detail.t0.confidence)} confidence · {detail.t0.review_priority.toFixed(2)} priority</span></div><div className="change-arrow"><span>↓</span><p>{detail.changes.new_evidence_records} new records<br /><em>{signed(detail.changes.p_review_warranted_change)} warranted probability</em></p></div><div className="phase-card current"><small>T1 · later evidence</small><strong>{labels[detail.t1.predicted_class]}</strong><span>{percent(detail.t1.confidence)} confidence · {detail.t1.review_priority.toFixed(2)} priority</span></div></> : <p className="muted-copy">Connect the API to compare phase snapshots.</p>}</article>
          <article className="panel human-review"><div className="panel-heading"><h3>Human review</h3><span>Demo workflow</span></div><p>These actions update the audit log only. They do not trigger enforcement.</p><button className="confirm" onClick={() => recordAction("confirm_for_further_review")}>Confirm for further review</button><button onClick={() => recordAction("dismiss_from_review")}>Dismiss from review</button><button onClick={() => recordAction("needs_more_information")}>Needs more information</button>{actionMessage && <div className="action-message">{actionMessage}</div>}</article>
          <article className="panel probability-panel"><div className="panel-heading"><h3>Class probabilities</h3><span>Sum: 1.00</span></div>{[["Review warranted", selected.p_review_warranted, "warranted"], ["Not warranted", selected.p_review_not_warranted, "not"], ["Insufficient evidence", selected.p_insufficient_evidence, "insufficient"]].map(([label, value, name]) => <div className="probability" key={label as string}><span>{label as string}</span><strong>{percent(value as number)}</strong><i><b className={name as string} style={{ width: percent(value as number) }} /></i></div>)}</article>
        </aside>
      </div>
    </section></div>}
  </main>;
}
