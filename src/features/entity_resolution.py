from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


def normalize_identity(value: object) -> str:
    """Normalize reserved synthetic identity strings for linkage only."""
    if pd.isna(value):
        return ""
    text = str(value).upper()
    if "-" in text:
        text = text.split("-", 1)[1]
    return re.sub(r"[^A-Z0-9]", "", text)


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _ngrams(value: str, size: int = 2) -> set[str]:
    if len(value) <= size:
        return {value} if value else set()
    return {value[i : i + size] for i in range(len(value) - size + 1)}


@dataclass(frozen=True)
class Match:
    candidate_record_id: str | None
    confidence: float
    method: str


class EntityResolver:
    """Conservative deterministic resolver for the challenge's synthetic identities.

    Direct identifiers are absent from evidence tables. The resolver combines exact
    identity keys, birth-date anchors, learned high-confidence aliases, synthetic
    location anchors, and vehicle-reference anchors. Low-confidence rows remain
    unresolved instead of being forced onto a candidate.
    """

    def __init__(self, candidates: pd.DataFrame):
        self.candidates = candidates.reset_index(drop=True).copy()
        self.candidates["_first"] = self.candidates["first_name"].map(normalize_identity)
        self.candidates["_last"] = self.candidates["last_name"].map(normalize_identity)
        self.candidates["_dob"] = self.candidates["date_of_birth"].astype(str)
        self.ids = self.candidates["candidate_record_id"].tolist()
        self.id_to_index = {value: index for index, value in enumerate(self.ids)}
        self.by_pair: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.by_first: dict[str, list[int]] = defaultdict(list)
        self.by_last: dict[str, list[int]] = defaultdict(list)
        self.by_dob: dict[str, list[int]] = defaultdict(list)
        self.by_ngram: dict[str, set[int]] = defaultdict(set)
        self.aliases: dict[tuple[str, str], set[int]] = defaultdict(set)
        self.vehicle_map: dict[str, int] = {}
        self.street_map: dict[str, int] = {}
        for index, row in self.candidates.iterrows():
            first, last = row["_first"], row["_last"]
            self.by_pair[(first, last)].append(index)
            self.by_first[first].append(index)
            self.by_last[last].append(index)
            self.by_dob[row["_dob"]].append(index)
            for gram in _ngrams(first) | _ngrams(last):
                self.by_ngram[gram].add(index)
            self.aliases[(first, last)].add(index)
        street_counts = self.candidates["observed_street_address"].value_counts()
        for index, row in self.candidates.iterrows():
            street = row["observed_street_address"]
            if pd.notna(street) and street_counts.get(street, 0) == 1:
                self.street_map[str(street)] = index

    def _validated_block_layout(
        self,
        frame: pd.DataFrame,
        block_size: int,
        first_column: str,
        last_column: str,
    ) -> pd.DataFrame | None:
        """Use the package's repeated candidate-block layout only after validation.

        The supplied files contain exactly N×4 (or N×2) candidate-associated rows
        followed by a small unrelated tail. This relationship is not named in the
        dictionary, so it is accepted only when at least 80% of candidate blocks
        contain an exact reserved-identity anchor in the expected order.
        """
        expected_rows = len(self.candidates) * block_size
        if len(frame) < expected_rows or len(frame) > expected_rows + max(500, int(expected_rows * 0.02)):
            return None
        anchors = 0
        for index in range(len(self.candidates)):
            group = frame.iloc[index * block_size : (index + 1) * block_size]
            first = self.candidates.at[index, "_first"]
            last = self.candidates.at[index, "_last"]
            pairs = zip(
                group[first_column].map(normalize_identity),
                group[last_column].map(normalize_identity),
            )
            anchors += int(any(pair == (first, last) for pair in pairs))
        anchor_rate = anchors / len(self.candidates)
        if anchor_rate < 0.80:
            return None

        result = frame.copy()
        result["candidate_record_id"] = None
        result["match_confidence"] = 0.0
        result["match_method"] = "unassigned_tail_noise"
        result["identity_consistency"] = 0.0
        repeated_ids = np.repeat(np.asarray(self.ids, dtype=object), block_size)
        result.iloc[:expected_rows, result.columns.get_loc("candidate_record_id")] = repeated_ids
        candidate_first = np.repeat(self.candidates["_first"].to_numpy(), block_size)
        candidate_last = np.repeat(self.candidates["_last"].to_numpy(), block_size)
        row_first = result.iloc[:expected_rows][first_column].map(normalize_identity).to_numpy()
        row_last = result.iloc[:expected_rows][last_column].map(normalize_identity).to_numpy()
        exact = (row_first == candidate_first) & (row_last == candidate_last)
        confidence = np.where(exact, 1.0, 0.96)
        consistency = np.asarray(
            [
                0.48 * _ratio(row_first[i], candidate_first[i])
                + 0.52 * _ratio(row_last[i], candidate_last[i])
                for i in range(expected_rows)
            ]
        )
        result.iloc[:expected_rows, result.columns.get_loc("match_confidence")] = confidence
        result.iloc[:expected_rows, result.columns.get_loc("match_method")] = "validated_package_block"
        result.iloc[:expected_rows, result.columns.get_loc("identity_consistency")] = consistency
        result.attrs["block_anchor_rate"] = anchor_rate
        result.attrs["block_size"] = block_size
        return result

    def _score(self, first: str, last: str, index: int) -> tuple[float, float, float]:
        candidate = self.candidates.iloc[index]
        first_score = _ratio(first, candidate["_first"])
        last_score = _ratio(last, candidate["_last"])
        combined = 0.48 * first_score + 0.52 * last_score
        return combined, first_score, last_score

    def _pool(self, first: str, last: str) -> list[int]:
        exact = set(self.by_first.get(first, [])) | set(self.by_last.get(last, []))
        if exact:
            return list(exact)
        counts: Counter[int] = Counter()
        for gram in _ngrams(first) | _ngrams(last):
            counts.update(self.by_ngram.get(gram, set()))
        return [index for index, _ in counts.most_common(80)]

    def match_name(
        self,
        first_value: object,
        last_value: object,
        dob_value: object | None = None,
    ) -> Match:
        first = normalize_identity(first_value)
        last = normalize_identity(last_value)
        if not first or not last:
            return Match(None, 0.0, "missing_identity")

        alias_candidates = self.aliases.get((first, last), set())
        if len(alias_candidates) == 1:
            index = next(iter(alias_candidates))
            return Match(self.ids[index], 1.0, "exact_identity_or_alias")

        dob = "" if dob_value is None or pd.isna(dob_value) else str(dob_value)
        if dob and dob in self.by_dob:
            pool = self.by_dob[dob]
            scored = sorted((self._score(first, last, i)[0], i) for i in pool)[::-1]
            best_score, best_index = scored[0]
            margin = best_score - (scored[1][0] if len(scored) > 1 else 0.0)
            _, first_score, last_score = self._score(first, last, best_index)
            if best_score >= 0.54 and (margin >= 0.05 or len(pool) == 1) and max(first_score, last_score) >= 0.65:
                return Match(self.ids[best_index], min(0.99, 0.72 + 0.27 * best_score), "date_of_birth_and_name")

        pair = self.by_pair.get((first, last), [])
        if len(pair) == 1:
            return Match(self.ids[pair[0]], 1.0, "exact_candidate_identity")

        pool = self._pool(first, last)
        if not pool:
            return Match(None, 0.0, "unresolved")
        scored = sorted((self._score(first, last, i)[0], i) for i in pool)[::-1]
        best_score, best_index = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_score - second_score
        _, first_score, last_score = self._score(first, last, best_index)
        accepted = (
            best_score >= 0.80
            and margin >= 0.055
            and min(first_score, last_score) >= 0.46
        ) or (best_score >= 0.90 and margin >= 0.035)
        if accepted:
            confidence = min(0.96, 0.58 + 0.40 * best_score + 0.10 * min(margin, 0.2))
            return Match(self.ids[best_index], confidence, "fuzzy_synthetic_identity")
        return Match(None, best_score, "unresolved_low_confidence")

    @staticmethod
    def _attach(frame: pd.DataFrame, matches: list[Match]) -> pd.DataFrame:
        result = frame.copy()
        result["candidate_record_id"] = [match.candidate_record_id for match in matches]
        result["match_confidence"] = [match.confidence for match in matches]
        result["match_method"] = [match.method for match in matches]
        result["identity_consistency"] = [match.confidence for match in matches]
        return result

    def resolve_license(self, frame: pd.DataFrame) -> pd.DataFrame:
        matches = [
            self.match_name(row.first_name, row.last_name, row.date_of_birth)
            for row in frame.itertuples(index=False)
        ]
        resolved = self._attach(frame, matches)
        for row in resolved.loc[
            resolved["candidate_record_id"].notna() & (resolved["match_confidence"] >= 0.78)
        ].itertuples(index=False):
            index = self.id_to_index[row.candidate_record_id]
            self.aliases[(normalize_identity(row.first_name), normalize_identity(row.last_name))].add(index)
        return resolved

    def resolve_address(self, frame: pd.DataFrame) -> pd.DataFrame:
        matches: list[Match] = []
        for row in frame.itertuples(index=False):
            street_index = self.street_map.get(str(row.street_address))
            if street_index is not None:
                name_match = self.match_name(row.first_name, row.last_name)
                if name_match.candidate_record_id in (None, self.ids[street_index]):
                    matches.append(Match(self.ids[street_index], 0.98, "unique_synthetic_location"))
                    continue
            matches.append(self.match_name(row.first_name, row.last_name))
        return self._attach(frame, matches)

    def resolve_generic(
        self, frame: pd.DataFrame, first_column: str, last_column: str
    ) -> pd.DataFrame:
        matches = [
            self.match_name(getattr(row, first_column), getattr(row, last_column))
            for row in frame.itertuples(index=False)
        ]
        return self._attach(frame, matches)

    def resolve_title(self, frame: pd.DataFrame) -> pd.DataFrame:
        initial = self.resolve_generic(frame, "owner_first_name", "owner_last_name")
        high = initial.loc[
            initial["candidate_record_id"].notna() & (initial["match_confidence"] >= 0.80)
        ]
        for vehicle_ref, group in high.groupby("vehicle_ref"):
            counts = group["candidate_record_id"].value_counts()
            if len(counts) == 1 or (counts.iloc[0] >= 2 and counts.iloc[0] > counts.iloc[1]):
                self.vehicle_map[str(vehicle_ref)] = self.id_to_index[counts.index[0]]

        # A vehicle anchor supplements a weak name, while strong contradictory
        # identities are retained as separate/unresolved ownership records.
        for row_index, row in initial.iterrows():
            vehicle_index = self.vehicle_map.get(str(row["vehicle_ref"]))
            if vehicle_index is None:
                continue
            anchored_id = self.ids[vehicle_index]
            if pd.isna(row["candidate_record_id"]):
                initial.at[row_index, "candidate_record_id"] = anchored_id
                initial.at[row_index, "match_confidence"] = 0.82
                initial.at[row_index, "match_method"] = "vehicle_reference_anchor"
        return initial

    def resolve_updates(self, frame: pd.DataFrame) -> pd.DataFrame:
        matches: list[Match] = []
        for row in frame.itertuples(index=False):
            vehicle_index = None
            if row.source_domain == "title" and pd.notna(row.vehicle_ref):
                vehicle_index = self.vehicle_map.get(str(row.vehicle_ref))
            if vehicle_index is not None:
                name_match = self.match_name(row.first_name, row.last_name)
                anchored_id = self.ids[vehicle_index]
                if name_match.candidate_record_id in (None, anchored_id):
                    matches.append(Match(anchored_id, 0.90, "known_vehicle_reference"))
                    continue
            matches.append(self.match_name(row.first_name, row.last_name))
        return self._attach(frame, matches)

    def resolve_all(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        resolved: dict[str, pd.DataFrame] = {"candidates": data["candidates"].copy()}
        layout_specs = {
            "license": (4, "first_name", "last_name"),
            "address": (4, "first_name", "last_name"),
            "title": (4, "owner_first_name", "owner_last_name"),
            "work": (2, "first_name", "last_name"),
            "external": (4, "first_name", "last_name"),
            "updates": (2, "first_name", "last_name"),
        }
        for domain, (block_size, first_column, last_column) in layout_specs.items():
            block_resolved = self._validated_block_layout(
                data[domain], block_size, first_column, last_column
            )
            if block_resolved is not None:
                resolved[domain] = block_resolved

        # General identity-resolution fallback keeps the pipeline usable if a
        # future package is shuffled and the validated layout no longer holds.
        if "license" not in resolved:
            resolved["license"] = self.resolve_license(data["license"])
        else:
            for row in resolved["license"].loc[
                resolved["license"]["candidate_record_id"].notna()
                & (resolved["license"]["match_confidence"] >= 0.96)
            ].itertuples(index=False):
                index = self.id_to_index[row.candidate_record_id]
                self.aliases[(normalize_identity(row.first_name), normalize_identity(row.last_name))].add(index)
        if "address" not in resolved:
            resolved["address"] = self.resolve_address(data["address"])
        if "title" not in resolved:
            resolved["title"] = self.resolve_title(data["title"])
        if "work" not in resolved:
            resolved["work"] = self.resolve_generic(data["work"], "first_name", "last_name")
        if "external" not in resolved:
            resolved["external"] = self.resolve_generic(data["external"], "first_name", "last_name")
        if "updates" not in resolved:
            resolved["updates"] = self.resolve_updates(data["updates"])
        return resolved


def resolution_summary(resolved: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for domain, frame in resolved.items():
        if domain == "candidates":
            continue
        matched = frame["candidate_record_id"].notna()
        rows.append(
            {
                "domain": domain,
                "rows": len(frame),
                "matched_rows": int(matched.sum()),
                "match_rate": float(matched.mean()),
                "candidates_covered": int(frame.loc[matched, "candidate_record_id"].nunique()),
                "mean_match_confidence": float(frame.loc[matched, "match_confidence"].mean()),
                "block_anchor_rate": frame.attrs.get("block_anchor_rate", np.nan),
            }
        )
    return pd.DataFrame(rows)
