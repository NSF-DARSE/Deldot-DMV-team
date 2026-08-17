"""Checks case_predictions.csv against every requirement in Submission_Format.md."""
import csv
from decimal import Decimal

import pandas as pd

from pipeline import BASE

PROBS = ["p_review_warranted", "p_review_not_warranted", "p_insufficient_evidence"]
VALID = {"review_warranted", "review_not_warranted", "insufficient_evidence"}
PATH = "case_predictions.csv"


def main():
    sub = pd.read_csv(PATH)
    tpl = pd.read_csv(BASE + "Submission_Template.csv")
    ok = True

    def check(label, condition, detail=""):
        nonlocal ok
        ok &= bool(condition)
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}{'  ' + detail if detail else ''}")

    print("Schema")
    check("column names and order match the template", list(sub.columns) == list(tpl.columns))
    check("row count matches the template", len(sub) == len(tpl), f"{len(sub)} rows")
    check("candidate/phase pairs match the template, in order",
          (sub[["candidate_record_id", "phase"]].values
           == tpl[["candidate_record_id", "phase"]].values).all())
    check("exactly one T0 and one T1 row per case",
          sub.groupby("candidate_record_id").phase.apply(
              lambda s: sorted(s) == ["T0", "T1"]).all())
    check("no missing values", int(sub.isna().sum().sum()) == 0)

    print("Values")
    check("predicted_class uses only the three allowed labels",
          set(sub.predicted_class) <= VALID)
    check("probabilities within [0, 1]",
          bool(((sub[PROBS] >= 0) & (sub[PROBS] <= 1)).all().all()))
    check("review_priority within [0, 1]",
          bool(((sub.review_priority >= 0) & (sub.review_priority <= 1)).all()),
          f"min {sub.review_priority.min():.4f} max {sub.review_priority.max():.4f}")
    # decimal, not binary float: the values as written to the file must sum to 1
    bad = 0
    with open(PATH) as f:
        for row in csv.DictReader(f):
            if sum(Decimal(row[c]) for c in PROBS) != 1:
                bad += 1
    check("each row's probabilities sum to exactly 1 as written", bad == 0,
          f"{bad} rows off")
    check("predicted_class is the argmax of the probabilities",
          (sub.predicted_class
           == sub[PROBS].idxmax(axis=1).str.replace("p_", "", regex=False)).all())

    print("\nDistribution")
    print(pd.crosstab(sub.phase, sub.predicted_class).to_string())
    w = sub.pivot(index="candidate_record_id", columns="phase", values="predicted_class")
    print(f"\nT0 -> T1 class changes: {(w.T0 != w.T1).mean():.1%} of cases")
    print(pd.crosstab(w.T0, w.T1).to_string())
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
