# Pre-Bridge Baseline Snapshot

This compact snapshot contains the linkage summary, linkage diagnostics, T0/T1 out-of-fold predictions, T0 fold assignments, and final OOF metrics from linkage-v1.0.0 before the conservative T0 vehicle-reference bridge was added.

It is included only to make the final package's paired before/after comparison reproducible. It is not the canonical model output. The canonical v1.1 outputs are under the top-level `outputs/` directory.

From the package root, rerun the comparison with:

```bash
.venv/bin/python scripts/compare_vehicle_bridge_v1.py
```
