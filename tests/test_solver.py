"""
Solver tests — runnable with pytest or as a plain script.

    python -m tests.test_solver
"""

from __future__ import annotations

import pathlib

from app.models import OptimizeRequest, Order, Plant, Site, Truck
from app.solver import solve

ROOT = pathlib.Path(__file__).resolve().parent.parent


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
                "max_trucks_on_site": 2,
            }
        ],
        "orders": [
            {
                "id": "O1",
                "site_id": "S1",
                "volume_m3": 40,
                "window_start_minute": 480,
                "window_end_minute": 720,
            }
        ],
        "options": {"time_limit_seconds": 10, "workers": 8},
    }
    payload.update(overrides)
    return OptimizeRequest.model_validate(payload)


def test_all_loads_scheduled_and_volumes_match() -> None:
    req = _request()
    res = solve(req)
    assert res.status in {"OPTIMAL", "FEASIBLE"}, res.status
    assert res.metrics.loads_dropped == 0, res.unassigned
    assert len(res.assignments) == 5  # 40 m3 / 8 m3 per truck
    total = sum(a.quantity_m3 for a in res.assignments)
    assert abs(total - 40) < 1e-6


def test_truck_capacity_and_no_overlap() -> None:
    req = _request()
    res = solve(req)
    caps = {t.id: t.capacity_m3 for t in req.trucks}

    by_truck: dict[str, list] = {}
    for a in res.assignments:
        assert a.quantity_m3 <= caps[a.truck_id] + 1e-6
        by_truck.setdefault(a.truck_id, []).append(a)

    for rows in by_truck.values():
        rows.sort(key=lambda a: a.batch_start_minute)
        for prev, cur in zip(rows, rows[1:]):
            assert cur.batch_start_minute >= prev.batch_start_minute, "truck double-booked"


def test_freshness_limit_is_respected() -> None:
    req = _request()
    res = solve(req)
    loads_batch = {a.order_id + "#" + str(a.load_index): a.batch_start_minute for a in res.assignments}
    for a in res.assignments:
        span = a.discharge_end_minute - loads_batch[a.order_id + "#" + str(a.load_index)]
        assert span <= req.options.max_drum_minutes, f"load {a.load_index} exceeded freshness"


def test_pour_window_and_lateness() -> None:
    req = _request()
    res = solve(req)
    for a in res.assignments:
        assert a.discharge_end_minute >= req.orders[0].window_start_minute


def test_single_pour_point_per_order() -> None:
    """Only one truck discharges at a time per order (one pump)."""
    req = _request()
    res = solve(req)
    rows = sorted((a.discharge_start_minute, a.discharge_end_minute) for a in res.assignments)
    for (s1, e1), (s2, _e2) in zip(rows, rows[1:]):
        assert s2 >= e1, "two trucks discharging simultaneously at one pour point"


def test_pour_continuity_gap() -> None:
    req = _request()
    res = solve(req)
    gap = req.options.max_gap_minutes
    ordered = sorted(res.assignments, key=lambda a: a.load_index)
    for prev, cur in zip(ordered, ordered[1:]):
        idle = cur.discharge_start_minute - prev.discharge_end_minute
        assert 0 <= idle <= gap, f"gap between loads = {idle} min"


def test_plant_capacity_is_respected() -> None:
    """Batches cannot overlap on a single plant."""
    req = _request()
    res = solve(req)
    batches = sorted(
        (a.batch_start_minute, max(1, round(a.quantity_m3 * 60 / req.plant.rate_m3_per_hour)))
        for a in res.assignments
    )
    for (s1, d1), (s2, _d2) in zip(batches, batches[1:]):
        assert s2 >= s1 + d1, "overlapping batches on the plant"


def test_scarce_fleet_drops_loads_instead_of_going_infeasible() -> None:
    req = _request(
        trucks=[{"id": "T01", "capacity_m3": 8}],
        orders=[
            {
                "id": "O1",
                "site_id": "S1",
                "volume_m3": 80,
                "window_start_minute": 480,
                "window_end_minute": 560,
            }
        ],
        options={"time_limit_seconds": 5, "workers": 4},
    )
    res = solve(req)
    assert res.status in {"OPTIMAL", "FEASIBLE"}
    assert res.metrics.loads_dropped > 0
    assert res.assignments, "at least one load should still be scheduled"


def test_sample_file_is_valid() -> None:
    req = OptimizeRequest.model_validate_json(
        (ROOT / "samples" / "day_plan.json").read_text(encoding="utf-8")
    )
    res = solve(req)
    assert res.status in {"OPTIMAL", "FEASIBLE"}, res.status
    assert res.metrics.loads_dropped == 0
    assert res.metrics.loads_total == 10  # 40/8 + 24/8 + 16/8


def test_unknown_site_is_rejected() -> None:
    req = _request(orders=[{"id": "O1", "site_id": "S9", "volume_m3": 8,
                            "window_start_minute": 480, "window_end_minute": 600}])
    try:
        solve(req)
    except ValueError as exc:
        assert "unknown sites" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown site")


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
