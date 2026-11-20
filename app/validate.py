"""
Independent plan validator — re-checks a finished plan against every hard
constraint, *without* trusting the solver.

Why this exists
---------------
The CP-SAT model is the source of the plan, but production needs a separate
guard because:

  * a plan can come from the greedy fallback (solver timed out),
  * a plan can come from a human edit (dispatcher drags a truck),
  * a model bug must never reach the plant floor silently.

`validate_plan(request, response)` returns a list of violations with codes that
map 1:1 to the constraints in `docs/DISPATCH_MATH.md`.

Usage
-----
    from .validate import validate_plan
    report = validate_plan(request, response)
    if not report.ok:
        raise ValueError(report.violations[0].message)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .models import Assignment, OptimizeRequest, OptimizeResponse

_EPS = 1e-6


@dataclass
class Violation:
    code: str
    message: str
    order_id: str | None = None
    load_index: int | None = None
    truck_id: str | None = None


@dataclass
class ValidationReport:
    ok: bool
    violations: list[Violation] = field(default_factory=list)
    checked: dict[str, int] = field(default_factory=dict)

    def add(self, code: str, message: str, **kw) -> None:
        self.violations.append(Violation(code=code, message=message, **kw))
        self.ok = False


def _discharge_minutes(rate_m3_per_hour: float, quantity: float) -> int:
    return max(1, int(math.ceil(quantity * 60.0 / rate_m3_per_hour - 1e-9)))


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Half-open intervals: [start, end). Touching is allowed."""
    return a_start < b_end and b_start < a_end


def validate_plan(req: OptimizeRequest, res: OptimizeResponse) -> ValidationReport:
    report = ValidationReport(ok=True)

    trucks = {t.id: t for t in req.trucks if t.active}
    sites = {s.id: s for s in req.sites}
    orders = {o.id: o for o in req.orders}
    load_time = req.plant.load_time_minutes
    max_late = req.options.max_late_minutes
    max_drum = req.options.max_drum_minutes

    # ── 0) References ──────────────────────────────────────────────────────
    for a in res.assignments:
        if a.order_id not in orders:
            report.add("UNKNOWN_ORDER", f"assignment references unknown order {a.order_id}")
        if a.truck_id not in trucks:
            report.add(
                "UNKNOWN_TRUCK",
                f"assignment references unknown/inactive truck {a.truck_id}",
                order_id=a.order_id,
                load_index=a.load_index,
                truck_id=a.truck_id,
            )

    checked = {
        "assignments": len(res.assignments),
        "orders": len(orders),
        "trucks": len(trucks),
    }
    if report.violations:
        report.checked = checked
        return report

    # ── 1) Load coverage: every ordered m3 is either scheduled or reported ──
    per_order = defaultdict(list)
    for a in res.assignments:
        per_order[a.order_id].append(a)

    unassigned_by_order = defaultdict(float)
    for u in res.unassigned:
        unassigned_by_order[u.order_id] += u.quantity_m3

    for order_id, order in orders.items():
        scheduled = sum(a.quantity_m3 for a in per_order.get(order_id, []))
        dropped = unassigned_by_order.get(order_id, 0.0)
        if abs(scheduled + dropped - order.volume_m3) > 0.05:
            report.add(
                "VOLUME_MISMATCH",
                f"order {order_id}: {scheduled:.2f} scheduled + {dropped:.2f} dropped "
                f"!= {order.volume_m3:.2f} ordered",
                order_id=order_id,
            )
        if dropped < -_EPS:
            report.add("NEGATIVE_DROP", f"order {order_id}: negative dropped volume")

    # ── 2) Truck capacity ──────────────────────────────────────────────────
    for a in res.assignments:
        truck = trucks[a.truck_id]
        if a.quantity_m3 > truck.capacity_m3 + _EPS:
            report.add(
                "TRUCK_CAPACITY",
                f"load {a.order_id}#{a.load_index} = {a.quantity_m3:.2f} m3 "
                f"exceeds truck {truck.code or truck.id} capacity {truck.capacity_m3:.2f}",
                order_id=a.order_id,
                load_index=a.load_index,
                truck_id=a.truck_id,
            )
        # ── 3) Truck availability window ───────────────────────────────────
        if a.batch_start_minute < truck.available_from_minute:
            report.add(
                "TRUCK_NOT_AVAILABLE",
                f"truck {truck.code or truck.id} batches at {a.batch_start_minute} "
                f"before shift start {truck.available_from_minute}",
                order_id=a.order_id,
                load_index=a.load_index,
                truck_id=a.truck_id,
            )
        if a.return_to_plant_minute > truck.available_to_minute:
            report.add(
                "TRUCK_OVERTIME",
                f"truck {truck.code or truck.id} returns at {a.return_to_plant_minute} "
                f"after shift end {truck.available_to_minute}",
                order_id=a.order_id,
                load_index=a.load_index,
                truck_id=a.truck_id,
            )

    # ── 4) Travel / duration consistency ───────────────────────────────────
    for a in res.assignments:
        site = sites[orders[a.order_id].site_id]
        expected_arrival = a.batch_start_minute + load_time + site.travel_from_plant_min
        if a.arrival_minute < expected_arrival - _EPS:
            report.add(
                "ARRIVAL_TOO_EARLY",
                f"load {a.order_id}#{a.load_index} arrives at {a.arrival_minute} "
                f"but cannot before {expected_arrival}",
                order_id=a.order_id,
                load_index=a.load_index,
            )
        if a.discharge_start_minute < a.arrival_minute:
            report.add(
                "DISCHARGE_BEFORE_ARRIVAL",
                f"load {a.order_id}#{a.load_index} discharges before arriving",
                order_id=a.order_id,
                load_index=a.load_index,
            )
        if a.discharge_end_minute < a.discharge_start_minute:
            report.add(
                "BAD_DISCHARGE_INTERVAL",
                f"load {a.order_id}#{a.load_index} discharge end < start",
                order_id=a.order_id,
                load_index=a.load_index,
            )
        expected_return = a.discharge_end_minute + site.return_minutes
        if a.return_to_plant_minute < expected_return - _EPS:
            report.add(
                "RETURN_TOO_EARLY",
                f"load {a.order_id}#{a.load_index} returns at "
                f"{a.return_to_plant_minute} but needs {expected_return}",
                order_id=a.order_id,
                load_index=a.load_index,
            )

    # ── 5) Freshness (hard): batching -> end of discharge ──────────────────
    for a in res.assignments:
        span = a.discharge_end_minute - a.batch_start_minute
        if span > max_drum + _EPS:
            report.add(
                "FRESHNESS_LIMIT",
                f"load {a.order_id}#{a.load_index}: {span} min from batching to "
                f"end of discharge > max_drum {max_drum}",
                order_id=a.order_id,
                load_index=a.load_index,
            )

    # ── 6) Pour window (start hard, end soft within max_late) ──────────────
    for a in res.assignments:
        order = orders[a.order_id]
        if a.discharge_start_minute < order.window_start_minute - _EPS:
            report.add(
                "BEFORE_WINDOW",
                f"load {a.order_id}#{a.load_index} pours at {a.discharge_start_minute} "
                f"before window start {order.window_start_minute}",
                order_id=a.order_id,
                load_index=a.load_index,
            )
        lateness = a.discharge_end_minute - order.window_end_minute
        if lateness > max_late + _EPS:
            report.add(
                "WINDOW_EXCEEDED",
                f"load {a.order_id}#{a.load_index} ends {lateness:.0f} min after the "
                f"window, over the allowed {max_late}",
                order_id=a.order_id,
                load_index=a.load_index,
            )

    # ── 7) Truck no-overlap + 8) plant no-overlap ──────────────────────────
    def _check_no_overlap(
        items: list[tuple[int, int, str, Assignment]],
        code: str,
        label: str,
    ) -> None:
        items.sort(key=lambda x: (x[0], x[1], x[2]))
        for prev, cur in zip(items, items[1:]):
            if _overlaps(prev[0], prev[1], cur[0], cur[1]):
                report.add(
                    code,
                    f"{label} {prev[2]}: {prev[3].order_id}#{prev[3].load_index} "
                    f"[{prev[0]},{prev[1]}) overlaps {cur[3].order_id}#{cur[3].load_index} "
                    f"[{cur[0]},{cur[1]})",
                    order_id=cur[3].order_id,
                    load_index=cur[3].load_index,
                    truck_id=cur[3].truck_id if code == "TRUCK_OVERLAP" else None,
                )

    per_truck: dict[str, list[tuple[int, int, str, Assignment]]] = defaultdict(list)
    for a in res.assignments:
        per_truck[a.truck_id].append((a.batch_start_minute, a.return_to_plant_minute, a.truck_id, a))
    for tid, items in per_truck.items():
        _check_no_overlap(items, "TRUCK_OVERLAP", f"truck {trucks[tid].code or tid}")

    site_occ: dict[str, list[tuple[int, int, str, Assignment]]] = {}
    for a in res.assignments:
        site = sites[orders[a.order_id].site_id]
        site_occ.setdefault(site.id, []).append(
            (a.arrival_minute, a.discharge_end_minute, site.id, a)
        )

    batches = [
        (a.batch_start_minute, a.batch_start_minute + _batch_minutes(req, a), a.truck_id, a)
        for a in res.assignments
    ]
    _check_no_overlap(batches, "PLANT_BATCH_OVERLAP", "plant batch")

    # ── 9) One pour point per order ────────────────────────────────────────
    for order_id, items in per_order.items():
        intervals = sorted(
            (a.discharge_start_minute, a.discharge_end_minute, order_id, i, a)
            for i, a in enumerate(items)
        )
        for prev, cur in zip(intervals, intervals[1:]):
            if _overlaps(prev[0], prev[1], cur[0], cur[1]):
                report.add(
                    "POUR_POINT_OVERLAP",
                    f"order {order_id}: loads #{prev[4].load_index} and #{cur[4].load_index} "
                    f"discharge at the same pour point simultaneously",
                    order_id=order_id,
                    load_index=cur[4].load_index,
                )    # ── 10) Site occupancy limit (Cumulative) ──────────────────────────────
    for site_id, items in site_occ.items():
        cap = sites[site_id].max_trucks_on_site
        events: list[tuple[int, int]] = []
        for start, end, _sid, _a in items:
            events.append((start, 1))
            events.append((end, -1))
        events.sort(key=lambda e: (e[0], e[1]))  # departures before arrivals at t
        current = 0
        for minute, delta in events:
            current += delta
            if current > cap:
                report.add(
                    "SITE_CAPACITY",
                    f"site {site_id}: {current} trucks on site at minute {minute} "
                    f"(limit {cap})",
                )
                break  # one message per site is enough

    # ── 11) Sequence sanity per truck ──────────────────────────────────────
    for tid, items in per_truck.items():
        seqs = [a.sequence for _s, _e, _t, a in items]
        if len(set(seqs)) != len(seqs):
            report.add(
                "DUPLICATE_SEQUENCE",
                f"truck {tid}: duplicate sequence numbers {sorted(seqs)}",
                truck_id=tid,
            )

    report.checked = checked
    return report


def _batch_minutes(req: OptimizeRequest, a: Assignment) -> int:
    """Minutes of plant time one load occupies (affine with its volume)."""
    return max(1, int(math.ceil(a.quantity_m3 * 60.0 / req.plant.rate_m3_per_hour - 1e-9)))
