Overview
========

The package links synthetic Delaware DMV evidence, prepares time-aware
features, and produces calibrated T0/T1 review predictions.

The official submission artifact is ``case_predictions.csv`` at the
repository root. Required columns:

* ``candidate_record_id``
* ``phase`` (``T0`` or ``T1``)
* ``predicted_class`` (``review_warranted``, ``review_not_warranted``, or ``insufficient_evidence``)
* ``p_review_warranted``, ``p_review_not_warranted``, ``p_insufficient_evidence`` (sum to 1)
* ``review_priority`` (0–1)

Every challenge case has exactly one T0 row and one T1 row.
