from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .normalization import clean_text, normalize_address, normalize_dob, normalize_name
from .similarity import jaro_winkler


@dataclass(frozen=True)
class LinkDecision:
    candidate_record_id: str | None
    link_confidence: float
    link_tier: str
    link_method: str
    first_name_similarity: float
    last_name_similarity: float
    name_similarity: float
    dob_match: int
    address_match: int
    candidate_pool_size: int
    runner_up_margin: float
    ambiguity_reason: str


class CandidateIndex:
    def __init__(self, candidates: pd.DataFrame, rules: dict):
        self.rules = rules
        self.weights = rules["name_weights"]
        self.thresholds = rules["thresholds"]
        self.confidence = rules["confidence_scores"]
        self.frame = candidates.copy().reset_index(drop=True)
        self.frame["_first"] = self.frame["first_name"].map(normalize_name)
        self.frame["_last"] = self.frame["last_name"].map(normalize_name)
        self.frame["_dob"] = self.frame["date_of_birth"].map(normalize_dob)
        self.frame["_address"] = self.frame["observed_street_address"].map(normalize_address)
        self.ids = self.frame["candidate_record_id"].tolist()

        self.by_name: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.by_dob: dict[str, list[int]] = defaultdict(list)
        self.by_address: dict[str, list[int]] = defaultdict(list)
        self.by_first: dict[str, list[int]] = defaultdict(list)
        self.by_last: dict[str, list[int]] = defaultdict(list)
        self.by_first_prefix: dict[str, set[int]] = defaultdict(set)
        self.by_last_prefix: dict[str, set[int]] = defaultdict(set)
        for idx, row in self.frame.iterrows():
            first, last = row["_first"], row["_last"]
            self.by_name[(first, last)].append(idx)
            self.by_first[first].append(idx)
            self.by_last[last].append(idx)
            if row["_dob"]:
                self.by_dob[row["_dob"]].append(idx)
            if row["_address"]:
                self.by_address[row["_address"]].append(idx)
            if first:
                self.by_first_prefix[first[:3]].add(idx)
            if last:
                self.by_last_prefix[last[:4]].add(idx)

        self.anchor_aliases: dict[tuple[str, str], set[int]] = defaultdict(set)

    def _name_scores(self, first: str, last: str, idx: int) -> tuple[float, float, float]:
        row = self.frame.iloc[idx]
        first_score = jaro_winkler(first, row["_first"])
        last_score = jaro_winkler(last, row["_last"])
        combined = self.weights["first"] * first_score + self.weights["last"] * last_score
        return combined, first_score, last_score

    def _rank(self, first: str, last: str, pool: Iterable[int]) -> list[tuple[float, float, float, int]]:
        ranked = []
        for idx in set(pool):
            combined, first_score, last_score = self._name_scores(first, last, idx)
            ranked.append((combined, first_score, last_score, idx))
        return sorted(ranked, reverse=True)

    def _decision(
        self,
        idx: int | None,
        confidence: float,
        tier: str,
        method: str,
        scores: tuple[float, float, float] = (0.0, 0.0, 0.0),
        *,
        dob_match: int = 0,
        address_match: int = 0,
        pool_size: int = 0,
        margin: float = 0.0,
        reason: str = "",
    ) -> LinkDecision:
        combined, first_score, last_score = scores
        return LinkDecision(
            self.ids[idx] if idx is not None else None,
            round(float(confidence), 6),
            tier,
            method,
            round(float(first_score), 6),
            round(float(last_score), 6),
            round(float(combined), 6),
            int(dob_match),
            int(address_match),
            int(pool_size),
            round(float(margin), 6),
            reason,
        )

    def resolve_dob_anchor(self, first_value: object, last_value: object, dob_value: object) -> LinkDecision:
        first, last, dob = normalize_name(first_value), normalize_name(last_value), normalize_dob(dob_value)
        if not first or not last:
            return self._decision(None, 0, "UNRESOLVED", "missing_name", reason="missing_first_or_last_name")
        if not dob or dob not in self.by_dob:
            return self._decision(None, 0, "UNRESOLVED", "dob_anchor_unavailable", reason="missing_or_unmatched_dob")

        pool = self.by_dob[dob]
        ranked = self._rank(first, last, pool)
        best, first_score, last_score, idx = ranked[0]
        margin = best - (ranked[1][0] if len(ranked) > 1 else 0.0)
        exact_name = first == self.frame.at[idx, "_first"] and last == self.frame.at[idx, "_last"]
        if exact_name:
            return self._decision(
                idx, self.confidence["exact_name_and_dob"], "A", "exact_name_and_dob",
                (best, first_score, last_score), dob_match=1, pool_size=len(pool), margin=margin,
            )

        t = self.thresholds
        if (
            best >= t["dob_name_combined_min"]
            and first_score >= t["dob_first_min"]
            and last_score >= t["dob_last_min"]
            and (len(pool) == 1 or margin >= t["dob_margin_min"])
        ):
            return self._decision(
                idx, self.confidence["fuzzy_name_and_dob"], "A", "fuzzy_name_and_exact_dob",
                (best, first_score, last_score), dob_match=1, pool_size=len(pool), margin=margin,
            )
        if (
            len(pool) == 1
            and best >= t["unique_dob_name_combined_min"]
            and first_score >= t["unique_dob_first_min"]
            and last_score >= t["unique_dob_last_min"]
        ):
            return self._decision(
                idx, self.confidence["unique_dob_and_acceptable_name"], "A", "unique_dob_and_acceptable_name",
                (best, first_score, last_score), dob_match=1, pool_size=1, margin=margin,
            )
        return self._decision(
            None, best, "UNRESOLVED", "dob_name_conflict", (best, first_score, last_score),
            dob_match=1, pool_size=len(pool), margin=margin, reason="dob_matches_but_name_is_weak_or_ambiguous",
        )

    def resolve_address_anchor(self, first_value: object, last_value: object, address_value: object) -> LinkDecision:
        first, last, address = normalize_name(first_value), normalize_name(last_value), normalize_address(address_value)
        if not first or not last:
            return self._decision(None, 0, "UNRESOLVED", "missing_name", reason="missing_first_or_last_name")
        if not address or address not in self.by_address:
            return self._decision(None, 0, "UNRESOLVED", "address_anchor_unavailable", reason="missing_or_unmatched_address")

        pool = self.by_address[address]
        ranked = self._rank(first, last, pool)
        best, first_score, last_score, idx = ranked[0]
        margin = best - (ranked[1][0] if len(ranked) > 1 else 0.0)
        t = self.thresholds
        if (
            len(pool) == 1
            and best >= t["address_name_combined_min"]
            and first_score >= t["address_first_min"]
            and last_score >= t["address_last_min"]
        ):
            return self._decision(
                idx, self.confidence["exact_address_and_name"], "B", "exact_unique_address_and_name",
                (best, first_score, last_score), address_match=1, pool_size=1, margin=margin,
            )
        if (
            len(pool) > 1
            and best >= t["ambiguous_address_name_combined_min"]
            and margin >= t["ambiguous_address_margin_min"]
        ):
            return self._decision(
                idx, self.confidence["ambiguous_address_resolved_by_name"], "B", "shared_address_resolved_by_name",
                (best, first_score, last_score), address_match=1, pool_size=len(pool), margin=margin,
            )
        return self._decision(
            None, best, "UNRESOLVED", "address_name_conflict", (best, first_score, last_score),
            address_match=1, pool_size=len(pool), margin=margin, reason="address_matches_but_name_is_weak_or_ambiguous",
        )

    def learn_anchor_aliases(self, rows: Iterable[tuple[object, object, LinkDecision]]) -> None:
        minimum = self.thresholds["alias_anchor_confidence_min"]
        id_to_idx = {candidate_id: idx for idx, candidate_id in enumerate(self.ids)}
        for first_value, last_value, decision in rows:
            if not decision.candidate_record_id or decision.link_confidence < minimum:
                continue
            first, last = normalize_name(first_value), normalize_name(last_value)
            if first and last:
                self.anchor_aliases[(first, last)].add(id_to_idx[decision.candidate_record_id])

    def _name_pool(self, first: str, last: str) -> set[int]:
        pool = set(self.by_first.get(first, ())) | set(self.by_last.get(last, ()))
        if not pool:
            pool |= self.by_first_prefix.get(first[:3], set()) if first else set()
            pool |= self.by_last_prefix.get(last[:4], set()) if last else set()
        return pool

    def resolve_name(self, first_value: object, last_value: object) -> LinkDecision:
        first, last = normalize_name(first_value), normalize_name(last_value)
        if not first or not last:
            return self._decision(None, 0, "UNRESOLVED", "missing_name", reason="missing_first_or_last_name")

        alias_pool = self.anchor_aliases.get((first, last), set())
        if len(alias_pool) == 1:
            idx = next(iter(alias_pool))
            scores = self._name_scores(first, last, idx)
            return self._decision(
                idx, self.confidence["verified_anchor_alias"], "B", "verified_dob_or_address_alias",
                scores, pool_size=1, margin=1.0,
            )
        if len(alias_pool) > 1:
            return self._decision(
                None, 0, "UNRESOLVED", "ambiguous_verified_alias", pool_size=len(alias_pool),
                reason="same_verified_alias_points_to_multiple_candidates",
            )

        exact_pool = self.by_name.get((first, last), [])
        if len(exact_pool) == 1:
            idx = exact_pool[0]
            return self._decision(
                idx, self.confidence["unique_exact_full_name"], "B", "unique_exact_normalized_full_name",
                (1.0, 1.0, 1.0), pool_size=1, margin=1.0,
            )
        if len(exact_pool) > 1:
            return self._decision(
                None, 1.0, "UNRESOLVED", "ambiguous_exact_full_name", (1.0, 1.0, 1.0),
                pool_size=len(exact_pool), reason="exact_full_name_is_not_unique",
            )

        t = self.thresholds
        if min(len(first), len(last)) < t["minimum_fuzzy_name_length"]:
            return self._decision(
                None, 0, "UNRESOLVED", "short_name_not_fuzzy_matched", reason="name_too_short_for_safe_fuzzy_linkage",
            )
        pool = self._name_pool(first, last)
        if not pool:
            return self._decision(None, 0, "UNRESOLVED", "no_name_block", reason="no_candidate_in_name_block")
        ranked = self._rank(first, last, pool)
        best, first_score, last_score, idx = ranked[0]
        margin = best - (ranked[1][0] if len(ranked) > 1 else 0.0)
        if (
            best >= t["name_only_combined_min"]
            and min(first_score, last_score) >= t["name_only_each_min"]
            and margin >= t["name_only_margin_min"]
        ):
            return self._decision(
                idx, self.confidence["strong_fuzzy_unique_name"], "C", "strong_fuzzy_name_clear_margin",
                (best, first_score, last_score), pool_size=len(pool), margin=margin,
            )
        return self._decision(
            None, best, "UNRESOLVED", "weak_or_ambiguous_fuzzy_name", (best, first_score, last_score),
            pool_size=len(pool), margin=margin, reason="fuzzy_name_below_strength_or_margin_threshold",
        )


class LinkagePipeline:
    SOURCE_SPECS = {
        "address_history": {
            "path": "Data_T0/address_history.csv", "phase": "T0", "first": "first_name", "last": "last_name",
            "state": "state", "event_date": "effective_start_date", "observed_date": "", "vehicle_ref": "",
        },
        "license_id_events": {
            "path": "Data_T0/license_id_events.csv", "phase": "T0", "first": "first_name", "last": "last_name",
            "state": "credential_state", "event_date": "event_date", "observed_date": "", "vehicle_ref": "",
        },
        "external_context_signals": {
            "path": "Data_T0/external_context_signals.csv", "phase": "T0", "first": "first_name", "last": "last_name",
            "state": "signal_state", "event_date": "effective_date", "observed_date": "", "vehicle_ref": "",
        },
        "vehicle_title_events": {
            "path": "Data_T0/vehicle_title_events.csv", "phase": "T0", "first": "owner_first_name", "last": "owner_last_name",
            "state": "event_state", "event_date": "event_date", "observed_date": "", "vehicle_ref": "vehicle_ref",
        },
        "work_location_signals": {
            "path": "Data_T0/work_location_signals.csv", "phase": "T0", "first": "first_name", "last": "last_name",
            "state": "work_state", "event_date": "observed_date", "observed_date": "observed_date", "vehicle_ref": "",
        },
        "evidence_update_stream": {
            "path": "Data_T1/evidence_update_stream.csv", "phase": "T1", "first": "first_name", "last": "last_name",
            "state": "state", "event_date": "effective_date", "observed_date": "observed_date", "vehicle_ref": "vehicle_ref",
        },
    }

    def __init__(self, data_root: Path, rules_path: Path):
        self.data_root = Path(data_root)
        self.rules_path = Path(rules_path)
        self.rules_bytes = self.rules_path.read_bytes()
        self.rules = json.loads(self.rules_bytes)
        self.rules_sha256 = hashlib.sha256(self.rules_bytes).hexdigest()
        candidates = pd.read_csv(self.data_root / "Data_T0/candidate_records.csv", dtype=str, keep_default_na=False)
        self.index = CandidateIndex(candidates, self.rules)
        self.candidates = candidates
        self.frames = {
            source: pd.read_csv(self.data_root / spec["path"], dtype=str, keep_default_na=False)
            for source, spec in self.SOURCE_SPECS.items()
        }
        self.decisions: dict[str, list[LinkDecision]] = {}
        self.vehicle_candidates: dict[str, set[str]] = defaultdict(set)

    @staticmethod
    def _map_unique(frame: pd.DataFrame, columns: list[str], function) -> list[LinkDecision]:
        keys = list(map(tuple, frame[columns].itertuples(index=False, name=None)))
        cache: dict[tuple, LinkDecision] = {}
        for key in dict.fromkeys(keys):
            cache[key] = function(*key)
        return [cache[key] for key in keys]

    def _build_anchors(self) -> None:
        license_frame = self.frames["license_id_events"]
        license_anchor = self._map_unique(
            license_frame, ["first_name", "last_name", "date_of_birth"], self.index.resolve_dob_anchor
        )
        address_frame = self.frames["address_history"]
        address_anchor = self._map_unique(
            address_frame, ["first_name", "last_name", "street_address"], self.index.resolve_address_anchor
        )
        anchor_rows = [
            (row.first_name, row.last_name, decision)
            for row, decision in zip(license_frame.itertuples(index=False), license_anchor)
        ] + [
            (row.first_name, row.last_name, decision)
            for row, decision in zip(address_frame.itertuples(index=False), address_anchor)
        ]
        self.index.learn_anchor_aliases(anchor_rows)
        self._license_anchors = license_anchor
        self._address_anchors = address_anchor

    def _prefer_anchor_or_name(self, anchors: list[LinkDecision], frame: pd.DataFrame, first: str, last: str) -> list[LinkDecision]:
        name_decisions = self._map_unique(frame, [first, last], self.index.resolve_name)
        return [anchor if anchor.candidate_record_id else name for anchor, name in zip(anchors, name_decisions)]

    def _build_vehicle_candidates(self) -> None:
        frame = self.frames["vehicle_title_events"]
        decisions = self.decisions["vehicle_title_events"]
        for vehicle_value, decision in zip(frame["vehicle_ref"], decisions):
            vehicle_ref = clean_text(vehicle_value)
            if vehicle_ref and decision.candidate_record_id and decision.link_tier in {"A", "B"}:
                self.vehicle_candidates[vehicle_ref].add(decision.candidate_record_id)

    def _candidate_name_margin(self, first_value: object, last_value: object, candidate_idx: int) -> float:
        first, last = normalize_name(first_value), normalize_name(last_value)
        candidate_score = self.index._name_scores(first, last, candidate_idx)[0]
        comparison_pool = self.index._name_pool(first, last)
        comparison_pool.discard(candidate_idx)
        if not comparison_pool:
            return candidate_score
        runner_up = max(self.index._name_scores(first, last, idx)[0] for idx in comparison_pool)
        return candidate_score - runner_up

    def _bridge_t0_vehicle_titles(self) -> None:
        """Recover only conservative T0 title links from vehicle_ref.

        Anchors are the pre-bridge, name-only title decisions. A vehicle is
        eligible only when every high-confidence anchor agrees and no other
        independently linked title owner points to a different candidate.
        The unresolved row's owner name must also strongly support the bridged
        candidate and beat the best alternate candidate by the frozen margin.
        """
        frame = self.frames["vehicle_title_events"]
        original = list(self.decisions["vehicle_title_events"])
        self._title_name_decisions = original
        thresholds = self.rules["thresholds"]
        anchor_min = float(thresholds["vehicle_bridge_anchor_confidence_min"])

        strong_anchors: dict[str, set[str]] = defaultdict(set)
        all_linked_owners: dict[str, set[str]] = defaultdict(set)
        strong_anchor_rows = 0
        for vehicle_value, decision in zip(frame["vehicle_ref"], original):
            vehicle_ref = clean_text(vehicle_value)
            if not vehicle_ref or not decision.candidate_record_id:
                continue
            all_linked_owners[vehicle_ref].add(decision.candidate_record_id)
            if decision.link_confidence >= anchor_min:
                strong_anchor_rows += 1
                strong_anchors[vehicle_ref].add(decision.candidate_record_id)

        strong_conflicts = {ref for ref, ids in strong_anchors.items() if len(ids) > 1}
        linked_owner_conflicts = {ref for ref, ids in all_linked_owners.items() if len(ids) > 1}
        bridge_candidates = {
            ref: next(iter(ids))
            for ref, ids in strong_anchors.items()
            if len(ids) == 1 and ref not in linked_owner_conflicts
        }
        self._t0_vehicle_bridge_candidates = bridge_candidates

        id_to_idx = {candidate_id: idx for idx, candidate_id in enumerate(self.index.ids)}
        bridged: list[LinkDecision] = []
        evaluated_unresolved = recovered = failed_name_safeguards = 0
        for row, decision in zip(frame.itertuples(index=False), original):
            if decision.candidate_record_id:
                bridged.append(decision)
                continue
            vehicle_ref = clean_text(row.vehicle_ref)
            candidate_id = bridge_candidates.get(vehicle_ref)
            if not candidate_id:
                bridged.append(decision)
                continue

            evaluated_unresolved += 1
            candidate_idx = id_to_idx[candidate_id]
            scores = self.index._name_scores(
                normalize_name(row.owner_first_name), normalize_name(row.owner_last_name), candidate_idx
            )
            combined, first_score, last_score = scores
            margin = self._candidate_name_margin(row.owner_first_name, row.owner_last_name, candidate_idx)
            if not (
                combined >= thresholds["vehicle_bridge_name_combined_min"]
                and first_score >= thresholds["vehicle_bridge_name_each_min"]
                and last_score >= thresholds["vehicle_bridge_name_each_min"]
                and margin >= thresholds["vehicle_bridge_name_margin_min"]
            ):
                failed_name_safeguards += 1
                bridged.append(decision)
                continue

            recovered += 1
            bridged.append(LinkDecision(
                candidate_id,
                self.rules["confidence_scores"]["unanimous_t0_vehicle_ref_and_strong_name"],
                "A",
                "unanimous_t0_vehicle_ref_with_strong_name",
                round(first_score, 6),
                round(last_score, 6),
                round(combined, 6),
                0,
                0,
                len(all_linked_owners.get(vehicle_ref, ())),
                round(margin, 6),
                "",
            ))

        self.decisions["vehicle_title_events"] = bridged
        self._t0_vehicle_bridge_diagnostics = {
            "anchor_definition": (
                "pre-bridge T0 title owner-name links with confidence >= "
                f"{anchor_min:.2f}; vehicle_ref is not used to establish anchors"
            ),
            "strong_anchor_rows": strong_anchor_rows,
            "vehicle_refs_with_strong_anchors": len(strong_anchors),
            "vehicle_refs_with_conflicting_strong_anchors": len(strong_conflicts),
            "vehicle_refs_with_any_linked_owner_conflict": len(linked_owner_conflicts),
            "vehicle_refs_eligible_for_bridge": len(bridge_candidates),
            "unresolved_rows_on_eligible_vehicle_refs": evaluated_unresolved,
            "rows_recovered": recovered,
            "rows_abstained_by_name_safeguards": failed_name_safeguards,
            "identity_inputs_used": ["vehicle_ref", "owner_first_name", "owner_last_name"],
            "identity_inputs_prohibited": ["row_order", "labels", "state", "event_date"],
        }

    def _resolve_update(self, row) -> LinkDecision:
        name_decision = self.index.resolve_name(row.first_name, row.last_name)
        if row.source_domain != "title" or not clean_text(row.vehicle_ref):
            return name_decision
        vehicle_pool = self.vehicle_candidates.get(clean_text(row.vehicle_ref), set())
        if len(vehicle_pool) != 1:
            if len(vehicle_pool) > 1:
                return LinkDecision(
                    None, name_decision.name_similarity, "UNRESOLVED", "ambiguous_vehicle_ownership",
                    name_decision.first_name_similarity, name_decision.last_name_similarity,
                    name_decision.name_similarity, 0, 0, len(vehicle_pool), 0.0,
                    "vehicle_ref_is_linked_to_multiple_candidates",
                )
            return name_decision
        vehicle_candidate = next(iter(vehicle_pool))
        if name_decision.candidate_record_id and name_decision.candidate_record_id != vehicle_candidate:
            return LinkDecision(
                None, name_decision.name_similarity, "UNRESOLVED", "vehicle_name_conflict",
                name_decision.first_name_similarity, name_decision.last_name_similarity,
                name_decision.name_similarity, 0, 0, 2, 0.0,
                "vehicle_anchor_and_name_link_point_to_different_candidates",
            )
        candidate_idx = self.index.ids.index(vehicle_candidate)
        scores = self.index._name_scores(normalize_name(row.first_name), normalize_name(row.last_name), candidate_idx)
        if scores[0] < self.rules["thresholds"]["vehicle_name_noncontradiction_min"]:
            return LinkDecision(
                None, scores[0], "UNRESOLVED", "vehicle_name_noncontradiction_failed",
                scores[1], scores[2], scores[0], 0, 0, 1, 0.0,
                "vehicle_ref_matches_but_owner_name_is_contradictory",
            )
        return LinkDecision(
            vehicle_candidate, self.rules["confidence_scores"]["known_vehicle_and_name"], "A",
            "known_vehicle_ref_with_noncontradictory_name", round(scores[1], 6), round(scores[2], 6),
            round(scores[0], 6), 0, 0, 1, 1.0, "",
        )

    def run(self) -> None:
        self._build_anchors()
        self.decisions["license_id_events"] = self._prefer_anchor_or_name(
            self._license_anchors, self.frames["license_id_events"], "first_name", "last_name"
        )
        self.decisions["address_history"] = self._prefer_anchor_or_name(
            self._address_anchors, self.frames["address_history"], "first_name", "last_name"
        )
        for source in ("external_context_signals", "work_location_signals"):
            frame = self.frames[source]
            self.decisions[source] = self._map_unique(frame, ["first_name", "last_name"], self.index.resolve_name)
        title = self.frames["vehicle_title_events"]
        self.decisions["vehicle_title_events"] = self._map_unique(
            title, ["owner_first_name", "owner_last_name"], self.index.resolve_name
        )
        self._bridge_t0_vehicle_titles()
        self._build_vehicle_candidates()
        updates = self.frames["evidence_update_stream"]
        keys = list(
            map(tuple, updates[["source_domain", "first_name", "last_name", "vehicle_ref"]].itertuples(index=False, name=None))
        )
        cache = {}
        for key in dict.fromkeys(keys):
            row = type("UpdateKey", (), dict(zip(["source_domain", "first_name", "last_name", "vehicle_ref"], key)))
            cache[key] = self._resolve_update(row)
        self.decisions["evidence_update_stream"] = [cache[key] for key in keys]

    def _standardize(self, source: str) -> pd.DataFrame:
        frame = self.frames[source]
        spec = self.SOURCE_SPECS[source]
        decision_frame = pd.DataFrame(asdict(decision) for decision in self.decisions[source])
        result = pd.DataFrame({
            "candidate_record_id": decision_frame["candidate_record_id"],
            "phase": spec["phase"],
            "source": source,
            "source_record_id": frame["source_record_id"],
            "source_domain": frame["source_domain"] if "source_domain" in frame else source,
            "event_date": frame[spec["event_date"]] if spec["event_date"] else "",
            "observed_date": frame[spec["observed_date"]] if spec["observed_date"] else "",
            "state": frame[spec["state"]] if spec["state"] else "",
            "vehicle_ref": frame[spec["vehicle_ref"]] if spec["vehicle_ref"] else "",
            "link_confidence": decision_frame["link_confidence"],
            "link_tier": decision_frame["link_tier"],
            "link_method": decision_frame["link_method"],
            "first_name_similarity": decision_frame["first_name_similarity"],
            "last_name_similarity": decision_frame["last_name_similarity"],
            "name_similarity": decision_frame["name_similarity"],
            "dob_match": decision_frame["dob_match"],
            "address_match": decision_frame["address_match"],
            "candidate_pool_size": decision_frame["candidate_pool_size"],
            "runner_up_margin": decision_frame["runner_up_margin"],
            "ambiguity_reason": decision_frame["ambiguity_reason"],
            "link_rule_version": self.rules["version"],
            "record_action": frame["record_action"] if "record_action" in frame else "",
            "event_type": frame["event_type"] if "event_type" in frame else "",
            "source_type": frame["source_type"] if "source_type" in frame else "",
            "signal_type": frame["signal_type"] if "signal_type" in frame else "",
            "evidence_quality": frame["evidence_quality"] if "evidence_quality" in frame else "",
            "credential_status": frame["credential_status"] if "credential_status" in frame else "",
            "effective_end_date": frame["effective_end_date"] if "effective_end_date" in frame else "",
            "source_description": frame["source_description"] if "source_description" in frame else "",
        })
        return result

    def outputs(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        standardized = pd.concat([self._standardize(source) for source in self.SOURCE_SPECS], ignore_index=True)
        linked = standardized[standardized["candidate_record_id"].notna()].copy()
        unresolved = standardized[standardized["candidate_record_id"].isna()].copy()

        summary_rows = []
        for (phase, source), group in standardized.groupby(["phase", "source"], sort=False):
            accepted = group["candidate_record_id"].notna()
            summary_rows.append({
                "phase": phase,
                "source": source,
                "total_records": len(group),
                "linked_records": int(accepted.sum()),
                "unresolved_records": int((~accepted).sum()),
                "link_rate": round(float(accepted.mean()), 6),
                "tier_a_records": int((group["link_tier"] == "A").sum()),
                "tier_b_records": int((group["link_tier"] == "B").sum()),
                "tier_c_records": int((group["link_tier"] == "C").sum()),
                "linked_candidate_coverage": int(group.loc[accepted, "candidate_record_id"].nunique()),
                "mean_link_confidence": round(float(group.loc[accepted, "link_confidence"].mean()), 6) if accepted.any() else 0.0,
            })
        summary = pd.DataFrame(summary_rows)

        methods = (
            standardized.groupby(["phase", "source", "link_tier", "link_method"], dropna=False)
            .size().rename("record_count").reset_index()
        )
        coverage = (
            linked.groupby(["candidate_record_id", "phase", "source"]).size().unstack([1, 2], fill_value=0)
        )
        coverage.columns = [f"{phase.lower()}__{source}" for phase, source in coverage.columns]
        coverage = self.candidates[["candidate_record_id"]].merge(
            coverage.reset_index(), on="candidate_record_id", how="left"
        ).fillna(0)
        for column in coverage.columns[1:]:
            coverage[column] = coverage[column].astype(int)
        return linked, unresolved, summary, methods, coverage

    def diagnostics(self) -> dict:
        all_decisions = [decision for decisions in self.decisions.values() for decision in decisions]
        linked = [decision for decision in all_decisions if decision.candidate_record_id]
        return {
            "link_rule_version": self.rules["version"],
            "rules_sha256": self.rules_sha256,
            "candidate_count": len(self.candidates),
            "source_record_count": len(all_decisions),
            "linked_record_count": len(linked),
            "unresolved_record_count": len(all_decisions) - len(linked),
            "overall_link_rate": round(len(linked) / len(all_decisions), 6),
            "candidate_name_key_collisions": sum(
                1 for indices in self.index.by_name.values() if len(indices) > 1
            ),
            "candidate_dob_key_collisions": sum(
                1 for indices in self.index.by_dob.values() if len(indices) > 1
            ),
            "candidate_address_key_collisions": sum(
                1 for indices in self.index.by_address.values() if len(indices) > 1
            ),
            "verified_alias_count": sum(
                1 for indices in self.index.anchor_aliases.values() if len(indices) == 1
            ),
            "ambiguous_verified_alias_count": sum(
                1 for indices in self.index.anchor_aliases.values() if len(indices) > 1
            ),
            "vehicle_ref_count": len(self.vehicle_candidates),
            "ambiguous_vehicle_ref_count": sum(
                1 for ids in self.vehicle_candidates.values() if len(ids) > 1
            ),
            "t0_vehicle_ref_bridge": self._t0_vehicle_bridge_diagnostics,
            "method_counts": dict(Counter(decision.link_method for decision in all_decisions)),
        }

    def vehicle_bridge_holdout_audit(self) -> dict:
        """Audit the frozen T0 bridge against independent strong name links.

        Each observed owner-name alias is removed in turn. The remaining
        anchors must satisfy the same unanimity rules, and the held-out name
        must satisfy the frozen name-strength and runner-up safeguards. This
        is internal consistency evidence, not authoritative link truth.
        """
        frame = self.frames["vehicle_title_events"].reset_index(drop=True)
        thresholds = self.rules["thresholds"]
        anchor_min = float(thresholds["vehicle_bridge_anchor_confidence_min"])
        rows = pd.DataFrame({
            "vehicle_ref": frame["vehicle_ref"].map(clean_text),
            "first": frame["owner_first_name"],
            "last": frame["owner_last_name"],
            "candidate_record_id": [decision.candidate_record_id for decision in self._title_name_decisions],
            "link_confidence": [decision.link_confidence for decision in self._title_name_decisions],
        })
        rows["alias"] = [
            f"{normalize_name(first)}|{normalize_name(last)}"
            for first, last in rows[["first", "last"]].itertuples(index=False, name=None)
        ]
        strong = rows[
            rows["candidate_record_id"].notna() & (rows["link_confidence"] >= anchor_min)
        ]
        linked = rows[rows["candidate_record_id"].notna()]
        id_to_idx = {candidate_id: idx for idx, candidate_id in enumerate(self.index.ids)}

        accepted = correct = incorrect = 0
        tested_refs: set[str] = set()
        for (vehicle_ref, alias), held_out in strong.groupby(["vehicle_ref", "alias"]):
            remaining_strong = strong[
                (strong["vehicle_ref"] == vehicle_ref) & (strong["alias"] != alias)
            ]
            strong_ids = set(remaining_strong["candidate_record_id"])
            if len(strong_ids) != 1:
                continue
            predicted = next(iter(strong_ids))
            remaining_linked = linked[
                (linked["vehicle_ref"] == vehicle_ref) & (linked["alias"] != alias)
            ]
            linked_ids = set(remaining_linked["candidate_record_id"])
            if linked_ids != {predicted}:
                continue
            predicted_idx = id_to_idx[predicted]
            for row in held_out.itertuples(index=False):
                combined, first_score, last_score = self.index._name_scores(
                    normalize_name(row.first), normalize_name(row.last), predicted_idx
                )
                margin = self._candidate_name_margin(row.first, row.last, predicted_idx)
                if not (
                    combined >= thresholds["vehicle_bridge_name_combined_min"]
                    and first_score >= thresholds["vehicle_bridge_name_each_min"]
                    and last_score >= thresholds["vehicle_bridge_name_each_min"]
                    and margin >= thresholds["vehicle_bridge_name_margin_min"]
                ):
                    continue
                accepted += 1
                tested_refs.add(vehicle_ref)
                if predicted == row.candidate_record_id:
                    correct += 1
                else:
                    incorrect += 1

        return {
            "reference_definition": "held-out pre-bridge T0 title links with confidence >= anchor threshold",
            "split_definition": "leave one normalized owner-name alias out within vehicle_ref",
            "accepted_reference_rows": accepted,
            "vehicle_refs_tested": len(tested_refs),
            "correct_against_reference": correct,
            "incorrect_against_reference": incorrect,
            "precision_against_reference": round(correct / accepted, 6) if accepted else None,
            "caveat": "Internal consistency audit only; no authoritative vehicle-owner link labels were supplied.",
        }

    def anchor_holdout_audit(self) -> dict:
        """Audit name-only linkage against strong DOB/name anchors without labels.

        The holdout split is by normalized observed-name pair, so the same alias
        cannot occur in both train and holdout. Exact DOB anchors are treated as
        internal reference links, not as authoritative ground truth.
        """
        license_frame = self.frames["license_id_events"]
        reference = []
        for row, decision in zip(license_frame.itertuples(index=False), self._license_anchors):
            if decision.candidate_record_id and decision.link_tier == "A":
                reference.append((row.first_name, row.last_name, decision))

        training, holdout = [], []
        for item in reference:
            key = f"{normalize_name(item[0])}|{normalize_name(item[1])}"
            bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 5
            (holdout if bucket == 0 else training).append(item)

        audit_index = CandidateIndex(self.candidates, self.rules)
        audit_index.learn_anchor_aliases(training)
        cache: dict[tuple[str, str], LinkDecision] = {}
        accepted = correct = incorrect = 0
        for first, last, reference_decision in holdout:
            key = (first, last)
            if key not in cache:
                cache[key] = audit_index.resolve_name(first, last)
            decision = cache[key]
            if not decision.candidate_record_id:
                continue
            accepted += 1
            if decision.candidate_record_id == reference_decision.candidate_record_id:
                correct += 1
            else:
                incorrect += 1

        return {
            "reference_definition": "accepted tier-A exact-DOB/name license links",
            "split_definition": "20% holdout by normalized observed-name alias hash",
            "reference_rows": len(reference),
            "holdout_rows": len(holdout),
            "accepted_holdout_rows": accepted,
            "correct_against_reference": correct,
            "incorrect_against_reference": incorrect,
            "precision_against_reference": round(correct / accepted, 6) if accepted else None,
            "coverage_on_reference_holdout": round(accepted / len(holdout), 6) if holdout else None,
            "caveat": "Internal consistency audit only; no authoritative link labels were supplied.",
        }
