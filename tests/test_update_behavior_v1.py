from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from oos_review.scripts.analyze_update_behavior_v1 import anchored_probabilities, evidence_groups


ROOT = Path(__file__).resolve().parents[1]


def test_anchoring_weights_preserve_or_blend_probabilities():
    t0 = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
    t1 = np.array([[0.2, 0.6, 0.2], [0.6, 0.2, 0.2], [0.6, 0.2, 0.2]])
    anchored = anchored_probabilities(t0, t1, np.array([1.0, 0.5, 0.0]))
    assert np.allclose(anchored[0], t1[0])
    assert np.allclose(anchored[1], 0.5 * t0[1] + 0.5 * t1[1])
    assert np.allclose(anchored[2], t0[2])
    assert np.allclose(anchored.sum(axis=1), 1.0)


def test_evidence_groups_partition_candidates():
    columns = pd.read_csv(
        ROOT / "data" / "outputs" / "feature_prep_v1" / "features_t1_delta.csv", nrows=0
    ).columns
    rows = [{column: 0 for column in columns} for _ in range(3)]
    rows[0]["new_t1_oos_count"] = 1
    rows[0]["new_t1_record_count"] = 1
    rows[1]["new_t1_record_count"] = 1
    frame = pd.DataFrame(rows)
    groups = evidence_groups(frame)
    assert groups.tolist() == [
        "meaningful_new_evidence",
        "weak_minor_update",
        "effectively_no_meaningful_update",
    ]


def test_behavior_outputs_retain_current_model_when_present():
    output = ROOT / "data" / "outputs" / "final_metrics_v1"
    if not output.exists():
        return
    diagnostics = json.loads((output / "update_behavior_diagnostics.json").read_text())
    behavior = pd.read_csv(output / "update_behavior_candidates.csv")
    comparison = pd.read_csv(output / "anchoring_comparison_oof.csv")
    assert len(behavior) == 12_000
    assert int(behavior.unanchored_class_changed.sum()) == diagnostics["total_unanchored_class_changes"]
    assert set(behavior.evidence_group) == {
        "meaningful_new_evidence",
        "weak_minor_update",
        "effectively_no_meaningful_update",
    }
    assert set(comparison.approach) == {"current_unanchored", "experimental_anchored"}
    assert diagnostics["decision"] == "retain_current_unanchored_t1"
    assert diagnostics["production_t1_predictions_modified"] is False
