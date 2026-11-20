"""
Validator tests — the independent guard must accept good plans and catch every
class of bad one.

    python -m tests.test_validate
"""

from __future__ import annotations

from app.models import Assignment, OptimizeRequest, OptimizeResponse
from app.solver import solve
from app.validate import validate_plan


def _request(**overrides) -> OptimizeRequest:
    payload = {
        "plan_date": "2026-11-21",
        "horizon_start_minute": 300,
        "horizon_end_minute": 1320,
        "plant": {"id": "P1", "rate_m3_per_hour": 120, "load_time_minutes": 5},
        "trucks": [
            {"id": f"T{i:02d}", "code": f"T{i:02d}", "capacity_m3": 8} for i in range(1, 7)
        ],
        "sites": [
            {
                "id": "S1",
                "travel_from_plant_min": 25,
                "travel_to_plant_min": 25,
                "discharge_rate_m3_per_hour": 20,
                "max_trucks_on_site": 1,
            }
        ],
        "orders": [
            {
                "id": "O1",
                "site_id": "S1",
                "volume_m3": 16,
                "window_start_minute": 480,
                "window_end_minute": 720,
            },
            {
                "id": "O2",
                "site_id": "S1",
                "volume_m3": 8,
                "window_start_minute": 480,
                "window_end_minute": 720,
            },
        ],
        "options": {"time_limit_seconds": 5, "workers": 8},
    }
    payload.update(overrides)
    return OptimizeRequest.model_validate(payload)


def _codes(report) -> set[str]:
    return {v.code for v in report.violations}


def test_validator_accepts_solver_plan() -> None:
    req = _request()
    res = solve(req)
    report = validate_plan(req, res)
    assert report.ok, [(v.code, v.message) for v in report.violations]


def test_validator_detects_truck_overlap() -> None:
    req = _request()
    res = solve(req)
    first = res.assignments[0]
    clash = res.assignments[-1].model_copy(
        update={
            "truck_id": first.truck_id,
            "batch_start_minute": first.batch_start_minute,
            "arrival_minute": first.arrival_minute,
            "discharge_start_minute": first.discharge_start_minute,
            "discharge_end_minute": first.discharge_end_minute,
            "return_to_plant_minute": first.return_to_plant_minute,
        }
    )
    broken = res.model_copy(update={"assignments": res.assignments[:-1] + [clash]})
    codes = _codes(validate_plan(req, broken))
    assert "TRUCK_OVERLAP" in codes, codes


def test_validator_detects_capacity_and_freshness() -> None:
    req = _request()
    res = solve(req)
    a = res.assignments[0]

    too_big = res.model_copy(
        update={"assignments": [a.model_copy(update={"quantity_m3": 20.0})] + res.assignments[1:]}
    )
    assert "TRUCK_CAPACITY" in _codes(validate_plan(req, too_big))

    stale = res.model_copy(
        update={
            "assignments": [
                a.model_copy(update={"batch_start_minute": a.discharge_end_minute - 200})
            ]
            + res.assignments[1:]
        }
    )
    assert "FRESHNESS_LIMIT" in _codes(validate_plan(req, stale))


def test_validator_detects_volume_mismatch() -> None:
    req = _request()
    res = solve(req)
    missing = res.model_copy(update={"assignments": res.assignments[1:]})
    assert "VOLUME_MISMATCH" in _codes(validate_plan(req, missing))


def test_validator_detects_pour_point_overlap() -> None:
    req = _request()
    res = solve(req)
    first = min(res.assignments, key=lambda x: (x.order_id, x.load_index))
    twin = next(
        a
        for a in res.assignments
        if a.order_id == first.order_id and a.load_index != first.load_index
    )
    clash = twin.model_copy(
        update={
            "discharge_start_minute": first.discharge_start_minute,
            "discharge_end_minute": first.discharge_end_minute,
        }
    )
    broken = res.model_copy(
        update={
            "assignments": [a if a is not twin else clash for a in res.assignments],
        }
    )
    assert "POUR_POINT_OVERLAP" in _codes(validate_plan(req, broken))


def test_validator_detects_site_capacity() -> None:
    req = _request()
    # Hand-built plan: two different orders discharging at the same moment on a
    # site that allows a single truck. Everything else is legal.
    assignments = [
        Assignment(
            order_id="O1",
            load_index=1,
            truck_id="T01",
            quantity_m3=8.0,
            batch_start_minute=400,
            arrival_minute=430,
            discharge_start_minute=430,
            discharge_end_minute=454,
            return_to_plant_minute=479,
            wait_minutes=0,
        ),
        Assignment(
            order_id="O2",
            load_index=1,
            truck_id="T02",
            quantity_m3=8.0,
            batch_start_minute=410,
            arrival_minute=435,
            discharge_start_minute=435,
            discharge_end_minute=459,
            return_to_plant_minute=484,
            wait_minutes=0,
        ),
    ]
    res = OptimizeResponse(status="FEASIBLE", solver_status="TEST", assignments=assignments)
    codes = _codes(validate_plan(req, res))
    assert "SITE_CAPACITY" in codes, codes


def test_validator_rejects_unknown_truck() -> None:
    req = _request()
    res = solve(req)
    bad = res.model_copy(
        update={"assignments": [res.assignments[0].model_copy(update={"truck_id": "T99"})]}
    )
    assert "UNKNOWN_TRUCK" in _codes(validate_plan(req, bad))


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"❌ {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
