"""Attach source-system rows to ``candidate_record_id``.

Flow
----
1. Build a ``PersonIndex`` from ``candidate_records.csv``.
2. For each source row, parse given / family / DOB.
3. Require an **exact** family-name match. Family prefixes are not used;
   ``ALCV`` and ``ALCVD`` are different people.
4. If the source row has a DOB, keep only candidates with that DOB.
   A DOB mismatch is a hard reject (same synthetic name can belong to
   different people).
5. Score the given name (exact, or truncation prefix).
6. If two or more candidates still survive, leave the row unassigned
   (``ambiguous_unassigned``). Do not guess.
7. Title / T1 leftover rows may inherit a unique vehicle owner only when
   the leftover name is compatible with that owner.

Why truncation is allowed on given names only
---------------------------------------------
About 12% of source given names are a single letter. Those are almost always
a truncated form of the candidate given name. With a matching DOB they
uniquely identify a candidate. Without a DOB they are too weak: some family
names have 50+ candidates.

Match rules written onto each linked row
----------------------------------------
``identity``
    Exact given + family + DOB.
``dob_prefix``
    Truncated given (overlap >= 3) + family + DOB.
``dob_initial``
    Truncated given (overlap 1–2) + family + DOB, unique among candidates.
``name_exact``
    Exact given + family; source has no DOB.
``name_prefix``
    Truncated given (overlap >= 3) + family; source has no DOB; unique.
``vehicle_ref``
    Second pass: vehicle already uniquely tied to one candidate, and this
    row's name is a truncation/exact of that owner. A different given name
    on the same title is treated as a different person.
``unlinked`` / ``ambiguous_unassigned``
    No unique candidate. The row is retained with a null id so coverage
    can be measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from oos_review.names import (
    GivenRelation,
    ParsedName,
    given_overlap_len,
    given_relation,
    parse_person,
)

# Minimum shared given-name characters required when the source has no DOB.
# Length 1–2 without DOB is rejected (see module docstring).
MIN_GIVEN_OVERLAP_WITHOUT_DOB = 3
MIN_GIVEN_OVERLAP_WITH_DOB = 1

SCORE_BY_RULE = {
    "identity": 1.00,
    "dob_prefix": 0.93,
    "dob_initial": 0.88,
    "name_exact": 0.86,
    "name_prefix": 0.72,
    "vehicle_ref": 0.80,
}


@dataclass(frozen=True)
class IndexedPerson:
    candidate_record_id: str
    given: str
    family: str
    dob: Optional[str]


@dataclass
class Match:
    candidate_record_id: str
    given_relation: GivenRelation
    overlap_len: int
    match_rule: str
    match_score: float
    n_matches: int = 1
    is_ambiguous: bool = False


class PersonIndex:
    """In-memory lookup of challenge candidates by family name and DOB."""

    def __init__(self, people: Iterable[IndexedPerson]):
        self.by_family: dict[str, list[IndexedPerson]] = {}
        self.by_family_dob: dict[tuple[str, str], list[IndexedPerson]] = {}
        self.by_id: dict[str, IndexedPerson] = {}
        for person in people:
            if not person.family or not person.candidate_record_id:
                continue
            self.by_id[person.candidate_record_id] = person
            self.by_family.setdefault(person.family, []).append(person)
            if person.dob:
                self.by_family_dob.setdefault(
                    (person.family, person.dob), []
                ).append(person)

    @classmethod
    def from_candidates(cls, candidates: pd.DataFrame) -> "PersonIndex":
        """Build the index from ``candidate_records.csv`` (or a subset)."""
        people: list[IndexedPerson] = []
        for row in candidates.itertuples(index=False):
            parsed = parse_person(
                getattr(row, "first_name"),
                getattr(row, "last_name"),
                getattr(row, "date_of_birth", None),
            )
            people.append(
                IndexedPerson(
                    candidate_record_id=str(
                        getattr(row, "candidate_record_id")
                    ),
                    given=parsed.given,
                    family=parsed.family,
                    dob=parsed.dob,
                )
            )
        return cls(people)

    def match(
        self,
        parsed: ParsedName,
        *,
        min_overlap_without_dob: int = MIN_GIVEN_OVERLAP_WITHOUT_DOB,
        min_overlap_with_dob: int = MIN_GIVEN_OVERLAP_WITH_DOB,
    ) -> Optional[Match]:
        """Return the unique best candidate, or ``None`` if unlinked/ambiguous."""
        winners = self.match_all(
            parsed,
            min_overlap_without_dob=min_overlap_without_dob,
            min_overlap_with_dob=min_overlap_with_dob,
        )
        if len(winners) != 1:
            return None
        return winners[0]

    def match_all(
        self,
        parsed: ParsedName,
        *,
        min_overlap_without_dob: int = MIN_GIVEN_OVERLAP_WITHOUT_DOB,
        min_overlap_with_dob: int = MIN_GIVEN_OVERLAP_WITH_DOB,
    ) -> list[Match]:
        """Return surviving candidate matches (0, 1, or many if ambiguous)."""
        if not parsed.family:
            return []

        if parsed.dob:
            pool = list(self.by_family_dob.get((parsed.family, parsed.dob), []))
            has_dob = True
        else:
            pool = list(self.by_family.get(parsed.family, []))
            has_dob = False

        scored: list[Match] = []
        min_overlap = (
            min_overlap_with_dob if has_dob else min_overlap_without_dob
        )
        for person in pool:
            relation = given_relation(parsed.given, person.given)
            if relation == "none":
                continue
            overlap = given_overlap_len(parsed.given, person.given)
            if relation != "exact" and overlap < min_overlap:
                continue
            rule = _rule_name(relation, overlap, has_dob)
            scored.append(
                Match(
                    candidate_record_id=person.candidate_record_id,
                    given_relation=relation,
                    overlap_len=overlap,
                    match_rule=rule,
                    match_score=SCORE_BY_RULE[rule],
                )
            )

        if not scored:
            return []

        exact = [m for m in scored if m.given_relation == "exact"]
        chosen = exact if exact else scored
        best_overlap = max(m.overlap_len for m in chosen)
        chosen = [m for m in chosen if m.overlap_len == best_overlap]
        for match in chosen:
            match.n_matches = len(chosen)
            match.is_ambiguous = len(chosen) > 1
        return chosen


def _rule_name(relation: GivenRelation, overlap: int, has_dob: bool) -> str:
    if has_dob and relation == "exact":
        return "identity"
    if has_dob and overlap >= 3:
        return "dob_prefix"
    if has_dob:
        return "dob_initial"
    if relation == "exact":
        return "name_exact"
    return "name_prefix"


_LINK_COLUMNS = [
    "candidate_record_id",
    "match_rule",
    "match_score",
    "given_relation",
    "overlap_len",
    "n_matches",
    "is_ambiguous",
]


def link_frame(
    source: pd.DataFrame,
    index: PersonIndex,
    *,
    first_col: str,
    last_col: str,
    dob_col: Optional[str] = None,
    drop_ambiguous: bool = True,
) -> pd.DataFrame:
    """Copy ``source`` and add linkage columns.

    Unlinked and ambiguous rows keep a null ``candidate_record_id`` so callers
    can measure coverage without silently dropping evidence. Ambiguous rows
    are labeled ``ambiguous_unassigned`` rather than guessed.
    """
    records: list[dict] = []
    for row in source.itertuples(index=False):
        dob_value = getattr(row, dob_col) if dob_col else None
        parsed = parse_person(
            getattr(row, first_col),
            getattr(row, last_col),
            dob_value,
        )
        matches = index.match_all(parsed)
        if len(matches) == 1:
            match = matches[0]
            records.append(
                {
                    "candidate_record_id": match.candidate_record_id,
                    "match_rule": match.match_rule,
                    "match_score": match.match_score,
                    "given_relation": match.given_relation,
                    "overlap_len": match.overlap_len,
                    "n_matches": match.n_matches,
                    "is_ambiguous": match.is_ambiguous,
                }
            )
        elif len(matches) > 1:
            _ = drop_ambiguous
            records.append(
                _empty_link("ambiguous_unassigned", n_matches=len(matches))
            )
        else:
            records.append(_empty_link("unlinked", n_matches=0))

    link_df = pd.DataFrame.from_records(records, columns=_LINK_COLUMNS)
    return pd.concat([source.reset_index(drop=True), link_df], axis=1)


def _empty_link(rule: str, n_matches: int) -> dict:
    return {
        "candidate_record_id": pd.NA,
        "match_rule": rule,
        "match_score": 0.0,
        "given_relation": "none",
        "overlap_len": 0,
        "n_matches": n_matches,
        "is_ambiguous": rule == "ambiguous_unassigned",
    }


def apply_vehicle_ref_pass(
    linked: pd.DataFrame,
    index: PersonIndex,
    *,
    first_col: str,
    last_col: str,
    vehicle_col: str = "vehicle_ref",
) -> pd.DataFrame:
    """Attach leftover rows when a vehicle already has exactly one owner.

    The vehicle id is extra evidence, not a license to ignore names. A row
    only inherits the unique owner if its family name matches that owner and
    its given name is exact or a truncation of the owner's given name.

    That lets ``SYNGIV-N`` on a uniquely owned vehicle attach, while
    ``SYNGIV-Uzlyyp`` on the same title (a different person) stays unlinked.
    Vehicles already tied to two candidates are left alone.
    """
    if vehicle_col not in linked.columns:
        return linked

    result = linked.copy()
    named = result[
        result["candidate_record_id"].notna()
        & result[vehicle_col].notna()
        & result["match_rule"].ne("vehicle_ref")
    ]
    vehicle_to_ids = named.groupby(vehicle_col)["candidate_record_id"].nunique()
    unique_vehicles = set(vehicle_to_ids[vehicle_to_ids == 1].index)
    vehicle_owner = (
        named[named[vehicle_col].isin(unique_vehicles)]
        .drop_duplicates(vehicle_col)
        .set_index(vehicle_col)["candidate_record_id"]
    )

    fill_idx: list[int] = []
    fill_ids: list[str] = []
    fill_rel: list[str] = []
    fill_overlap: list[int] = []
    for row in result.itertuples():
        if not pd.isna(row.candidate_record_id):
            continue
        vehicle = getattr(row, vehicle_col)
        if pd.isna(vehicle) or vehicle not in vehicle_owner.index:
            continue
        owner_id = str(vehicle_owner.loc[vehicle])
        owner = index.by_id.get(owner_id)
        if owner is None:
            continue
        parsed = parse_person(getattr(row, first_col), getattr(row, last_col))
        if parsed.family != owner.family:
            continue
        relation = given_relation(parsed.given, owner.given)
        if relation == "none":
            continue
        fill_idx.append(int(row.Index))
        fill_ids.append(owner_id)
        fill_rel.append(relation)
        fill_overlap.append(given_overlap_len(parsed.given, owner.given))

    if not fill_idx:
        return result

    result.loc[fill_idx, "candidate_record_id"] = fill_ids
    result.loc[fill_idx, "match_rule"] = "vehicle_ref"
    result.loc[fill_idx, "match_score"] = SCORE_BY_RULE["vehicle_ref"]
    result.loc[fill_idx, "given_relation"] = fill_rel
    result.loc[fill_idx, "overlap_len"] = fill_overlap
    result.loc[fill_idx, "n_matches"] = 1
    result.loc[fill_idx, "is_ambiguous"] = False
    return result


# Column map for each challenge file. dob_col is None when the file has none.
SOURCE_SPECS = {
    "address_history": {
        "first_col": "first_name",
        "last_col": "last_name",
        "dob_col": None,
        "vehicle_pass": False,
    },
    "license_id_events": {
        "first_col": "first_name",
        "last_col": "last_name",
        "dob_col": "date_of_birth",
        "vehicle_pass": False,
    },
    "vehicle_title_events": {
        "first_col": "owner_first_name",
        "last_col": "owner_last_name",
        "dob_col": None,
        "vehicle_pass": True,
    },
    "work_location_signals": {
        "first_col": "first_name",
        "last_col": "last_name",
        "dob_col": None,
        "vehicle_pass": False,
    },
    "external_context_signals": {
        "first_col": "first_name",
        "last_col": "last_name",
        "dob_col": None,
        "vehicle_pass": False,
    },
    "evidence_update_stream": {
        "first_col": "first_name",
        "last_col": "last_name",
        "dob_col": None,
        "vehicle_pass": True,
    },
}


def _link_one(
    frame: pd.DataFrame,
    index: PersonIndex,
    spec: dict,
) -> pd.DataFrame:
    linked = link_frame(
        frame,
        index,
        first_col=spec["first_col"],
        last_col=spec["last_col"],
        dob_col=spec["dob_col"],
    )
    if spec["vehicle_pass"]:
        linked = apply_vehicle_ref_pass(
            linked,
            index,
            first_col=spec["first_col"],
            last_col=spec["last_col"],
        )
    return linked


def link_t0_sources(
    candidates: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    *,
    index: Optional[PersonIndex] = None,
) -> dict[str, pd.DataFrame]:
    """Link every T0 source table. ``sources`` keys must be SOURCE_SPECS keys."""
    index = index or PersonIndex.from_candidates(candidates)
    linked: dict[str, pd.DataFrame] = {}
    for name, frame in sources.items():
        if name not in SOURCE_SPECS:
            raise KeyError(f"Unknown source '{name}'. Known: {list(SOURCE_SPECS)}")
        linked[name] = _link_one(frame, index, SOURCE_SPECS[name])
    return linked


def link_t1_stream(
    candidates: pd.DataFrame,
    updates: pd.DataFrame,
    *,
    index: Optional[PersonIndex] = None,
    title_linked: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Link T1 update rows. Optionally seed vehicle owners from T0 titles."""
    index = index or PersonIndex.from_candidates(candidates)
    spec = SOURCE_SPECS["evidence_update_stream"]
    linked = _link_one(updates, index, spec)
    if title_linked is None or "vehicle_ref" not in title_linked.columns:
        return linked

    # Seed T1 vehicle ownership from T0 title links, then re-run the vehicle
    # pass so a T1 title row with a truncated name can still attach.
    t0_named = title_linked[
        title_linked["candidate_record_id"].notna()
        & title_linked["vehicle_ref"].notna()
    ][["vehicle_ref", "candidate_record_id"]]
    if t0_named.empty:
        return linked

    combined = pd.concat(
        [
            linked,
            t0_named.assign(
                match_rule="identity",
                match_score=1.0,
                given_relation="exact",
                overlap_len=0,
                n_matches=1,
                is_ambiguous=False,
            ).reindex(columns=linked.columns),
        ],
        ignore_index=True,
    )
    combined = apply_vehicle_ref_pass(
        combined,
        index,
        first_col=spec["first_col"],
        last_col=spec["last_col"],
    )
    return combined.iloc[: len(linked)].copy()
