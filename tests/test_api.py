from backend.main import cases, health


def test_api_health_contract():
    payload = health()
    assert payload["status"] == "ok"
    assert payload["candidate_count"] == 12_000
    assert payload["prediction_rows"] == 24_000
    assert payload["decision_support_only"] is True


def test_cases_returns_priority_order():
    payload = cases(
        phase="T1",
        predicted_class=None,
        min_priority=0.0,
        max_priority=1.0,
        limit=10,
        offset=0,
    )
    priorities = [item["review_priority"] for item in payload["items"]]
    assert priorities == sorted(priorities, reverse=True)

