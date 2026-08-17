"""Run the current pipeline stage and optionally write artifacts.

Stage 1 (this module): name/DOB linkage.
Later stages will extend ``run_pipeline`` without changing call sites.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from oos_review import paths
from oos_review.linker import PersonIndex, link_t0_sources, link_t1_stream
from oos_review.load import load_candidates, load_t0_sources, load_t1_stream


def run_linkage(*, save: bool = True, output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Link T0 sources and the T1 update stream to candidate_record_id.

    Returns a dict that always includes ``candidates`` plus one frame per
    source. Linked frames keep original columns and add match metadata.
    """
    candidates = load_candidates()
    index = PersonIndex.from_candidates(candidates)
    t0_linked = link_t0_sources(candidates, load_t0_sources(), index=index)
    t1_linked = link_t1_stream(
        candidates,
        load_t1_stream(),
        index=index,
        title_linked=t0_linked["vehicle_title_events"],
    )

    bundle = {"candidates": candidates, **t0_linked, "evidence_update_stream": t1_linked}

    if save:
        dest = output_dir or paths.LINKED_DIR
        dest.mkdir(parents=True, exist_ok=True)
        for name, frame in bundle.items():
            frame.to_csv(dest / f"{name}.csv", index=False)

    return bundle
