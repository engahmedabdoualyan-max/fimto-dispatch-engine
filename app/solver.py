"""
CP-SAT model for Ready-Mix Concrete dispatching.

Decision
--------
For every *load* (an order split into truck-sized chunks) the model decides
when batching starts and which truck carries it, honouring:

  * plant batching capacity (no two batches at the same time)
  * truck capacity and availability window
  * travel time plant -> site -> plant
  * one pour point per order (one truck discharging at a time)
  * site occupancy limit (trucks present at the same time)
  * order pouring window (soft, penalised lateness)
  * concrete freshness (batching -> end of discharge, hard limit)
  * pour continuity (consecutive loads back-to-back, penalised gaps)

Objective (minimised, integers)
    drop*missed + late*minutes_late + wait*truck_waiting + gap*pour_gaps
    + truck*trucks_used + makespan*day_length

Scaling
-------
A naive model has one optional interval per (load × truck) pair, which is huge
and highly symmetric. Two mechanisms keep it fast:

1. `heuristic.py` builds a feasible plan in pure Python (milliseconds); it is
   injected as a CP-SAT *hint*, so the solver starts from a real plan.
2. Trucks with identical capacity/shift are interchangeable, so we add
   symmetry-breaking: used(T01) >= used(T02) >= ...
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from .heuristic import GreedyLoad, greedy_hints
from .models import (
    Assignment,
    Metrics,
    OptimizeRequest,
    OptimizeResponse,
    Unassigned,
    minute_to_hhmm,
)

_EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Internal load representation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Load:
    lid: str
    order_id: str
    index: int
    quantity: float
    site_id: str
    window_start: int
    window_end: int
    discharge_rate: float
    max_trucks_on_site: int
    candidates: list[str]
    reason: str | None = None

    # filled during model building
    start: cp_model.IntVar | None = None
    drop: cp_model.BoolVar | None = None
    present: cp_model.BoolVar | None = None
    discharge_start: cp_model.IntVar | None = None
    return_end: cp_model.IntVar | None = None
    wait: cp_model.IntVar | None = None
    late: cp_model.IntVar | None = None
    picks: list[cp_model.BoolVar] = field(default_factory=list)
    batch_minutes: int = 0
    discharge_minutes: int = 0


@dataclass
class _Timing:
    """Constant durations for one load (independent of the assigned truck)."""

    load_time: int
    travel_out: int
    travel_back: int
    discharge: int
    batch: int

    @property
    def freshness_span(self) -> int:
        """Minimum batching-start -> discharge-end duration."""
        return self.load_time + self.travel_out + self.discharge


@dataclass
class _Built:
    model: cp_model.CpModel
    loads: list[_Load]
    hints: dict[str, dict]
    truck_used: dict[str, cp_model.IntVar] = field(default_factory=dict)
    makespan: cp_model.IntVar | None = None


def _split_into_loads(volume: float, load_size: float) -> list[float]:
    """Split `volume` into full `load_size` loads plus one remainder."""
    if volume <= _EPS:
        return []
    size = max(load_size, 0.5)
    full = int(math.floor(volume / size + 1e-9))
    remainder = round(volume - full * size, 3)
    loads = [round(size, 3)] * full
    if remainder > 0.05:
        loads.append(remainder)
    return loads or [round(volume, 3)]


def _reference_load_size(req: OptimizeRequest) -> float:
    if req.options.reference_load_m3:
        return float(req.options.reference_load_m3)
    capacities = [round(t.capacity_m3, 2) for t in req.trucks if t.active]
    if not capacities:
        return 8.0
    # most common capacity wins (homogeneous fleets -> exact split)
    return Counter(capacities).most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────


def build_model(req: OptimizeRequest) -> _Built:
    plant = req.plant
    opts = req.options
    weights = opts.weights

    sites = {s.id: s for s in req.sites}
    trucks = {t.id: t for t in req.trucks if t.active}

    unknown_sites = [o.id for o in req.orders if o.site_id not in sites]
    if unknown_sites:
        raise ValueError(f"orders reference unknown sites: {', '.join(unknown_sites)}")

    load_size = _reference_load_size(req)
    h0, h1 = req.horizon_start_minute, req.horizon_end_minute
    ds_hi = max(h1, max((o.window_end_minute for o in req.orders), default=h1)) + opts.max_late_minutes

    loads: list[_Load] = []
    timings: dict[str, _Timing] = {}

    for order in req.orders:
        site = sites[order.site_id]
        discharge_rate = order.pour_rate_m3_per_hour or site.discharge_rate_m3_per_hour
        max_on_site = order.max_trucks_on_site or site.max_trucks_on_site

        for idx, qty in enumerate(_split_into_loads(order.volume_m3, load_size), start=1):
            lid = f"{order.id}#{idx}"
            timing = _Timing(
                load_time=plant.load_time_minutes,
                travel_out=site.travel_from_plant_min,
                travel_back=site.return_minutes,
                discharge=max(1, math.ceil(qty * 60 / discharge_rate)),
                batch=max(1, math.ceil(qty * 60 / plant.rate_m3_per_hour)),
            )
            timings[lid] = timing

            candidates: list[str] = []
            for truck in trucks.values():
                if truck.capacity_m3 + _EPS < qty:
                    continue
                # Time-compatibility pruning keeps the model small.
                earliest_ds = max(h0, truck.available_from_minute) + timing.load_time + timing.travel_out
                if earliest_ds + timing.discharge > order.window_end_minute + opts.max_late_minutes:
                    continue
                latest_ds = truck.available_to_minute - timing.travel_back - timing.discharge
                if latest_ds < order.window_start_minute:
                    continue
                candidates.append(truck.id)

            reason = None
            biggest = max((t.capacity_m3 for t in trucks.values()), default=0)
            if timing.freshness_span > opts.max_drum_minutes:
                reason, candidates = "FRESHNESS_LIMIT", []
            elif not candidates:
                reason = "FRESHNESS_LIMIT" if qty > biggest + _EPS else "NO_FEASIBLE_TRUCK"
            elif h1 - h0 < timing.freshness_span + timing.travel_back:
                reason, candidates = "TIME_WINDOW", []

            loads.append(
                _Load(
                    lid=lid,
                    order_id=order.id,
                    index=idx,
                    quantity=qty,
                    site_id=order.site_id,
                    window_start=order.window_start_minute,
                    window_end=order.window_end_minute,
                    discharge_rate=discharge_rate,
                    max_trucks_on_site=max_on_site,
                    candidates=candidates,
                    reason=reason,
                    batch_minutes=timing.batch,
                    discharge_minutes=timing.discharge,
                )
            )

    # ── Warm start from the greedy heuristic ────────────────────────────────
    greedy_input = [
        GreedyLoad(
            lid=l.lid,
            order_id=l.order_id,
            index=l.index,
            quantity=l.quantity,
            site_id=l.site_id,
            window_start=l.window_start,
            window_end=l.window_end,
            batch_minutes=l.batch_minutes,
            discharge_minutes=l.discharge_minutes,
            load_time=timings[l.lid].load_time,
            travel_out=timings[l.lid].travel_out,
            travel_back=timings[l.lid].travel_back,
        )
        for l in loads
        if l.candidates
    ]
    hints = (
        greedy_hints(req, greedy_input, {l.lid: set(l.candidates) for l in loads if l.candidates})
        if greedy_input
        else {}
    )

    model = cp_model.CpModel()
    schedulable = [l for l in loads if l.candidates]

    per_truck: dict[str, list[cp_model.IntervalVar]] = {tid: [] for tid in trucks}
    per_site_occ: dict[str, list[cp_model.IntervalVar]] = defaultdict(list)
    per_order_discharge: dict[str, list[cp_model.IntervalVar]] = defaultdict(list)
    batch_intervals: list[cp_model.IntervalVar] = []

    for load in schedulable:
        timing = timings[load.lid]
        start_hi = h1 - (timing.load_time + timing.travel_out + timing.discharge)
        if start_hi < h0:
            load.candidates = []
            load.reason = "TIME_WINDOW"
            continue

        start = model.NewIntVar(h0, start_hi, f"start_{load.lid}")
        drop = model.NewBoolVar(f"drop_{load.lid}")
        present = model.NewBoolVar(f"present_{load.lid}")
        model.Add(present + drop == 1)

        arrival = start + timing.load_time + timing.travel_out
        ds = model.NewIntVar(h0, ds_hi, f"ds_{load.lid}")
        model.Add(ds >= arrival)

        wait = model.NewIntVar(0, ds_hi - h0 + timing.travel_back, f"wait_{load.lid}")
        model.Add(wait == ds - arrival)

        # Truck cycle = load + travel out + on-site wait + discharge + travel back.
        # Kept affine in `wait` because CP-SAT interval sizes must be affine.
        cycle_size = wait + timing.load_time + timing.travel_out + timing.discharge + timing.travel_back
        return_end = model.NewIntVar(
            h0, ds_hi + timing.discharge + timing.travel_back, f"end_{load.lid}"
        )
        model.Add(return_end == start + cycle_size)

        late = model.NewIntVar(0, opts.max_late_minutes, f"late_{load.lid}")
        model.Add(ds >= load.window_start).OnlyEnforceIf(present)
        model.Add(ds + timing.discharge <= load.window_end + late)
        model.Add(late == 0).OnlyEnforceIf(drop)
        model.Add(wait == 0).OnlyEnforceIf(drop)

        # Concrete freshness: batching start -> end of discharge.
        model.Add(ds + timing.discharge - start <= opts.max_drum_minutes).OnlyEnforceIf(present)

        picks: list[cp_model.BoolVar] = []
        for tid in load.candidates:
            truck = trucks[tid]
            p = model.NewBoolVar(f"x_{load.lid}_{tid}")
            picks.append(p)
            model.Add(start >= truck.available_from_minute).OnlyEnforceIf(p)
            model.Add(return_end <= truck.available_to_minute).OnlyEnforceIf(p)
            per_truck[tid].append(
                model.NewOptionalIntervalVar(
                    start, cycle_size, return_end, p, f"cyc_{load.lid}_{tid}"
                )
            )

        model.Add(sum(picks) == present)

        # A truck occupies the site from arrival until it leaves; one pour
        # point per order (capacity 1) but several trucks may queue on site.
        per_site_occ[load.site_id].append(
            model.NewOptionalIntervalVar(
                arrival,
                wait + timing.discharge,  # affine -> valid interval size
                ds + timing.discharge,
                present,
                f"occ_{load.lid}",
            )
        )
        per_order_discharge[load.order_id].append(
            model.NewOptionalFixedSizeIntervalVar(
                ds, timing.discharge, present, f"disch_{load.lid}"
            )
        )
        batch_intervals.append(
            model.NewOptionalFixedSizeIntervalVar(
                start, timing.batch, present, f"batch_{load.lid}"
            )
        )

        load.start, load.drop, load.present = start, drop, present
        load.discharge_start, load.return_end = ds, return_end
        load.wait, load.late = wait, late
        load.picks = picks

    # ── Resource constraints ────────────────────────────────────────────────
    for intervals in per_truck.values():
        if intervals:
            model.AddNoOverlap(intervals)

    if batch_intervals:
        model.AddNoOverlap(batch_intervals)

    for site_id, intervals in per_site_occ.items():
        model.AddCumulative(intervals, [1] * len(intervals), sites[site_id].max_trucks_on_site)

    for intervals in per_order_discharge.values():
        # one pump / one pour point
        model.AddCumulative(intervals, [1] * len(intervals), 1)

    for order in req.orders:
        intervals = per_order_discharge.get(order.id, [])
        if intervals and order.max_trucks_on_site:
            model.AddCumulative(intervals, [1] * len(intervals), order.max_trucks_on_site)

    # ── Pour continuity (soft: penalise idle time between loads) ────────────
    order_loads: dict[str, list[_Load]] = defaultdict(list)
    for load in schedulable:
        order_loads[load.order_id].append(load)

    gap_terms: list = []
    for order_id, group in order_loads.items():
        ordered = sorted(group, key=lambda l: l.index)
        for prev, cur in zip(ordered, ordered[1:]):
            assert prev.discharge_start is not None and cur.discharge_start is not None
            both = [prev.present, cur.present]
            prev_end = prev.discharge_start + timings[prev.lid].discharge
            model.Add(cur.discharge_start >= prev.discharge_start).OnlyEnforceIf(both)
            if opts.max_gap_minutes is not None:
                over = model.NewIntVar(
                    0, ds_hi - h0 + 480, f"gap_{order_id}_{cur.index}"
                )
                model.Add(over >= cur.discharge_start - prev_end - opts.max_gap_minutes).OnlyEnforceIf(both)
                gap_terms.append(weights.gap * over)

    # ── Makespan & truck usage ──────────────────────────────────────────────
    makespan = model.NewIntVar(h0, ds_hi + 480, "makespan")
    for load in schedulable:
        if load.present is not None and load.return_end is not None:
            model.Add(makespan >= load.return_end).OnlyEnforceIf(load.present)

    truck_used: dict[str, cp_model.IntVar] = {}
    for tid in trucks:
        lits = [
            load.picks[i]
            for load in schedulable
            for i, c in enumerate(load.candidates)
            if c == tid
        ]
        used = model.NewBoolVar(f"used_{tid}")
        if lits:
            model.Add(sum(lits) <= len(lits) * used)
            model.Add(used <= sum(lits))
        else:
            model.Add(used == 0)
        truck_used[tid] = used

    # Symmetry breaking: identical trucks are interchangeable.
    groups: dict[tuple, list[str]] = defaultdict(list)
    for tid in sorted(trucks):
        t = trucks[tid]
        groups[(round(t.capacity_m3, 2), t.available_from_minute, t.available_to_minute, t.driver_id)].append(tid)
    for ids in groups.values():
        for prev, cur in zip(ids, ids[1:]):
            model.Add(truck_used[prev] >= truck_used[cur])

    # ── Objective ───────────────────────────────────────────────────────────
    terms: list = []
    for load in schedulable:
        terms.append(weights.drop * load.drop)
        terms.append(weights.late * load.late)
        terms.append(weights.wait * load.wait)
    terms.extend(gap_terms)
    terms.append(weights.makespan * makespan)
    for used in truck_used.values():
        terms.append(weights.truck * used)
    model.Minimize(sum(terms))

    # ── Hints (greedy plan) ─────────────────────────────────────────────────
    _apply_hints(model, schedulable, hints, makespan)

    return _Built(model=model, loads=loads, hints=hints, truck_used=truck_used, makespan=makespan)


def _apply_hints(
    model: cp_model.CpModel,
    loads: list[_Load],
    hints: dict[str, dict],
    makespan: cp_model.IntVar,
) -> None:
    """Feed the greedy plan to CP-SAT so it starts from a feasible solution."""
    latest_end = 0
    for load in loads:
        hint = hints.get(load.lid)
        if not hint or load.start is None:
            continue
        dropped = hint.get("drop") == 1
        model.AddHint(load.drop, 1 if dropped else 0)
        model.AddHint(load.present, 0 if dropped else 1)
        if dropped:
            continue
        model.AddHint(load.start, hint["start"])
        model.AddHint(load.discharge_start, hint["ds"])
        model.AddHint(load.wait, hint["wait"])
        model.AddHint(load.late, hint.get("late", 0))
        chosen = hint.get("truck")
        for tid, p in zip(load.candidates, load.picks):
            model.AddHint(p, 1 if tid == chosen else 0)
        if load.return_end is not None:
            latest_end = max(latest_end, hint["ds"] + load.discharge_minutes)
    if latest_end:
        model.AddHint(makespan, latest_end)


# ─────────────────────────────────────────────────────────────────────────────
# Solve + extract
# ─────────────────────────────────────────────────────────────────────────────


def solve(req: OptimizeRequest) -> OptimizeResponse:
    started = time.perf_counter()
    if not req.orders:
        return OptimizeResponse(
            status="OPTIMAL",
            solver_status="EMPTY",
            solve_ms=0,
            plan_date=req.plan_date,
            metrics=Metrics(),
        )

    built = build_model(req)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = req.options.time_limit_seconds
    solver.parameters.num_search_workers = req.options.workers
    solver.parameters.random_seed = req.options.seed
    status = solver.Solve(built.model) if _has_search_vars(built) else cp_model.OPTIMAL
    status_name = solver.StatusName(status) if _has_search_vars(built) else "OPTIMAL"

    planned = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assignments: list[Assignment] = []
    unassigned: list[Unassigned] = []

    if planned:
        assignments, unassigned = _extract_from_solver(req, built, solver)
    else:
        # Fallback: return the greedy plan rather than nothing.
        assignments, unassigned = _extract_from_hints(req, built)
        if assignments:
            status_name = "FEASIBLE"

    metrics = _metrics(req, assignments, unassigned, built, status, planned or bool(assignments))
    objective = None
    if planned:
        objective = int(round(solver.ObjectiveValue()))

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return OptimizeResponse(
        status=status_name if status_name in {"OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN", "MODEL_INVALID"} else "UNKNOWN",
        solver_status=status_name,
        objective=objective,
        solve_ms=elapsed_ms,
        plan_date=req.plan_date,
        assignments=assignments,
        unassigned=unassigned,
        trucks=_truck_summary(req, assignments),
        metrics=metrics,
        message=None
        if assignments
        else "No feasible plan found — falling back to manual dispatch.",
    )


def _has_search_vars(built: _Built) -> bool:
    return any(l.candidates for l in built.loads)


def _extract_from_solver(
    req: OptimizeRequest, built: _Built, solver: cp_model.CpSolver
) -> tuple[list[Assignment], list[Unassigned]]:
    assignments: list[Assignment] = []
    unassigned: list[Unassigned] = []

    for load in built.loads:
        if not load.candidates:
            unassigned.append(
                Unassigned(
                    order_id=load.order_id,
                    load_index=load.index,
                    quantity_m3=load.quantity,
                    reason=load.reason or "NO_FEASIBLE_TRUCK",  # type: ignore[arg-type]
                )
            )
            continue

        assert load.start is not None and load.drop is not None
        assert load.discharge_start is not None and load.wait is not None

        if solver.Value(load.drop) == 1:
            unassigned.append(
                Unassigned(
                    order_id=load.order_id,
                    load_index=load.index,
                    quantity_m3=load.quantity,
                    reason="CAPACITY_OR_CONGESTION",
                )
            )
            continue

        chosen = next(
            (tid for tid, p in zip(load.candidates, load.picks) if solver.Value(p) == 1), None
        )
        if chosen is None:  # defensive
            unassigned.append(
                Unassigned(
                    order_id=load.order_id,
                    load_index=load.index,
                    quantity_m3=load.quantity,
                    reason="NO_FEASIBLE_TRUCK",
                )
            )
            continue

        assignments.append(
            _assignment_from_minutes(
                req,
                load,
                chosen,
                solver.Value(load.start),
                solver.Value(load.discharge_start),
                solver.Value(load.wait),
            )
        )

    return _finalise(assignments, unassigned)


def _extract_from_hints(
    req: OptimizeRequest, built: _Built
) -> tuple[list[Assignment], list[Unassigned]]:
    assignments: list[Assignment] = []
    unassigned: list[Unassigned] = []

    for load in built.loads:
        hint = built.hints.get(load.lid)
        if not load.candidates or not hint or hint.get("drop") == 1:
            unassigned.append(
                Unassigned(
                    order_id=load.order_id,
                    load_index=load.index,
                    quantity_m3=load.quantity,
                    reason=load.reason or "CAPACITY_OR_CONGESTION",  # type: ignore[arg-type]
                )
            )
            continue
        assignments.append(
            _assignment_from_minutes(
                req,
                load,
                hint["truck"],
                hint["start"],
                hint["ds"],
                hint["wait"],
            )
        )

    return _finalise(assignments, unassigned)


def _assignment_from_minutes(
    req: OptimizeRequest,
    load: _Load,
    truck_id: str,
    batch: int,
    ds: int,
    wait: int,
) -> Assignment:
    site = next(s for s in req.sites if s.id == load.site_id)
    arrival = batch + req.plant.load_time_minutes + site.travel_from_plant_min
    return Assignment(
        order_id=load.order_id,
        load_index=load.index,
        truck_id=truck_id,
        truck_code=_truck_code(req, truck_id),
        quantity_m3=load.quantity,
        batch_start_minute=batch,
        arrival_minute=arrival,
        discharge_start_minute=ds,
        discharge_end_minute=ds + load.discharge_minutes,
        return_to_plant_minute=ds + load.discharge_minutes + site.return_minutes,
        wait_minutes=wait,
        batch_start=minute_to_hhmm(batch),
        arrival=minute_to_hhmm(arrival),
        discharge_end=minute_to_hhmm(ds + load.discharge_minutes),
    )


def _truck_code(req: OptimizeRequest, truck_id: str) -> str | None:
    return next((t.code for t in req.trucks if t.id == truck_id), None)


def _finalise(
    assignments: list[Assignment], unassigned: list[Unassigned]
) -> tuple[list[Assignment], list[Unassigned]]:
    by_truck: dict[str, list[Assignment]] = defaultdict(list)
    for a in assignments:
        by_truck[a.truck_id].append(a)
    for rows in by_truck.values():
        for seq, row in enumerate(sorted(rows, key=lambda r: r.batch_start_minute), start=1):
            row.sequence = seq
    assignments.sort(key=lambda a: (a.truck_id, a.batch_start_minute))
    return assignments, unassigned


def _truck_summary(req: OptimizeRequest, assignments: list[Assignment]) -> list[dict]:
    loads_per_truck: Counter[str] = Counter(a.truck_id for a in assignments)
    return [
        {
            "id": t.id,
            "code": t.code,
            "capacity_m3": t.capacity_m3,
            "loads": loads_per_truck.get(t.id, 0),
            "utilised": loads_per_truck.get(t.id, 0) > 0,
        }
        for t in sorted(req.trucks, key=lambda x: x.id)
    ]


def _metrics(
    req: OptimizeRequest,
    assignments: list[Assignment],
    unassigned: list[Unassigned],
    built: _Built,
    status: int,
    valid: bool,
) -> Metrics:
    if not valid:
        return Metrics(loads_total=len(built.loads), loads_dropped=len(unassigned))

    window_end = {o.id: o.window_end_minute for o in req.orders}
    late_minutes = sum(
        max(0, a.discharge_end_minute - window_end.get(a.order_id, a.discharge_end_minute))
        for a in assignments
    )
    orders_late = len(
        {
            a.order_id
            for a in assignments
            if a.discharge_end_minute > window_end.get(a.order_id, a.discharge_end_minute)
        }
    )
    total_wait = sum(a.wait_minutes for a in assignments)
    horizon = max(1, req.horizon_end_minute - req.horizon_start_minute)
    scheduled_batch = sum(
        max(1, math.ceil(a.quantity_m3 * 60 / req.plant.rate_m3_per_hour)) for a in assignments
    )
    makespan = max((a.return_to_plant_minute for a in assignments), default=req.horizon_start_minute)

    return Metrics(
        loads_total=len(built.loads),
        loads_scheduled=len(assignments),
        loads_dropped=len(unassigned),
        trucks_used=len({a.truck_id for a in assignments}),
        late_minutes=int(late_minutes),
        orders_late=orders_late,
        total_wait_minutes=int(total_wait),
        avg_wait_minutes=round(total_wait / len(assignments), 1) if assignments else 0.0,
        makespan_minutes=max(0, makespan - req.horizon_start_minute),
        plant_utilization=round(min(1.0, scheduled_batch / horizon), 3),
    )
