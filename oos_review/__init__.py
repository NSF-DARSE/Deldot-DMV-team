"""Decision-support package for the out-of-state tag holder review challenge.

Current stage
-------------
1. Record linkage by normalized name / DOB.
2. Per-case feature table and a transparent recency-vote rule baseline.

The end-to-end flow is documented in ``docs/DATA_FLOW.md``.
"""

from oos_review.baseline import apply_baseline, to_submission
from oos_review.caseview import dossier, linkage_summary
from oos_review.features import build_case_features, build_t0_t1_features
from oos_review.linker import PersonIndex, link_frame, link_t0_sources, link_t1_stream
from oos_review.names import ParsedName, parse_dob, parse_family, parse_given

__all__ = [
    "ParsedName",
    "PersonIndex",
    "apply_baseline",
    "build_case_features",
    "build_t0_t1_features",
    "dossier",
    "link_frame",
    "link_t0_sources",
    "link_t1_stream",
    "linkage_summary",
    "parse_dob",
    "parse_family",
    "parse_given",
    "to_submission",
]
