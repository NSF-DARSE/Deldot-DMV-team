"""Decision-support package for the out-of-state tag holder review challenge.

Current stage
-------------
Record linkage: attach source-system rows to ``candidate_record_id`` using
normalized given name, family name, and date of birth.

Later stages (features, T1 updates, classification) will live alongside this
package. The end-to-end flow is documented in ``docs/DATA_FLOW.md``.
"""

from oos_review.caseview import dossier, linkage_summary
from oos_review.linker import PersonIndex, link_frame, link_t0_sources, link_t1_stream
from oos_review.names import ParsedName, parse_dob, parse_family, parse_given

__all__ = [
    "ParsedName",
    "PersonIndex",
    "dossier",
    "link_frame",
    "link_t0_sources",
    "link_t1_stream",
    "linkage_summary",
    "parse_dob",
    "parse_family",
    "parse_given",
]

__all__ = [
    "ParsedName",
    "PersonIndex",
    "link_frame",
    "link_t0_sources",
    "link_t1_stream",
    "parse_dob",
    "parse_family",
    "parse_given",
]
