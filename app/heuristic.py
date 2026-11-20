"""
Greedy construction heuristic (event-driven list scheduling).

CP-SAT alone needs minutes to find a first solution on a 200-load day because
every load has one optional interval per truck (huge symmetry). We therefore
build a good feasible plan in pure Python first and hand it to the solver as a
hint. The solver then only has to *improve* it, which it does in a few seconds.

Why event-driven? Scheduling one order completely before the next keeps the
plant idle between orders and pushes later orders past their window. Instead we
always schedule the *most urgent next load* over all orders:

    repeat:
        for each order that still has loads:
            try to place its next load as early as possible
        pick the placement with the earliest discharge time
        commit it, update truck / plant / pour-point state

This mirrors what an experienced dispatcher does and keeps the plant busy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import OptimizeRequest


@dataclass
class GreedyLoad:
    lid: str
    order_id: str
    index: int
    quantity: float
    site_id: str
    window_start: int
    window_end: int
    batch_minutes: int
    discharge_minutes: int
    load_time: int
    travel_out: int
    travel_back: int

    @property
    def freshness_span(self) -> int:
        return self.load_time + self.travel_out + self.discharge_minutes


@dataclass
class _OrderState:
    order_id: str
    loads: list[GreedyLoad]
    cursor: int  # earliest minute the pour point is free
    next_index: int = 0
    dropped: int = 0

    @property
    def remaining(self) -> list[GreedyLoad]:
        return self.loads[self.next_index :]


@dataclass
class _Placement:
    ds: int
    order_id: str
    load: GreedyLoad
    truck_id: str
    batch: int
    wait: int
    ready: int = 0


def _plan(
    req: OptimizeRequest,
    loads: list[GreedyLoad],
    candidates: dict[str, set[str]],
) -> dict[str, dict]:
    trucks = {t.id: t for t in req.trucks if t.active}
    truck_free = {t.id: t.available_from_minute for t in trucks.values()}
    # One free-time per allowed concurrent truck slot (max_trucks_on_site).
    site_slots: dict[str, list[int]] = {}
    site_capacity = {s.id: s.max_trucks_on_site for s in req.sites}
    plant_free = req.horizon_start_minute
    max_late = req.options.max_late_minutes
    max_drum = req.options.max_drum_minutes

    states: list[_OrderState] = []
    for order in req.orders:
        order_loads = sorted((l for l in loads if l.order_id == order.id), key=lambda l: l.index)
        if not order_loads:
            continue
        states.append(_OrderState(order_id=order.id, loads=order_loads, cursor=order.window_start_minute))

    hints: dict[str, dict] = {}
    pending = [s for s in states if s.loads]

    while pending:
        best: _Placement | None = None

        for state in pending:
            load = state.loads[state.next_index]
            allowed = candidates.get(load.lid, set())
            begin = max(state.cursor, load.window_start)
            # Site occupancy: a truck can only discharge in a free slot, and the
            # site never hosts more than `max_trucks_on_site` trucks at once.
            slots = site_slots.setdefault(load.site_id, [])
            capacity = max(1, site_capacity.get(load.site_id, 1))
            if len(slots) >= capacity:
                slots.sort()
                begin = max(begin, slots[0])

            for tid, truck in trucks.items():
                if tid not in allowed:
                    continue
                if truck.capacity_m3 + 1e-6 < load.quantity:
                    continue
                if truck.available_to_minute - truck.available_from_minute < load.freshness_span:
                    continue
                if truck.available_to_minute - load.travel_back - load.discharge_minutes < load.window_start:
                    continue

                ready = max(truck_free[tid], plant_free)
                batch = max(ready, begin - load.load_time - load.travel_out)
                arrival = batch + load.load_time + load.travel_out
                ds = max(arrival, begin)
                wait = ds - arrival

                if ds + load.discharge_minutes > load.window_end + max_late:
                    continue
                if wait + load.load_time + load.travel_out + load.discharge_minutes > max_drum:
                    continue
                if ds + load.discharge_minutes + load.travel_back > truck.available_to_minute:
                    continue

                candidate = _Placement(
                    ds=ds,
                    order_id=state.order_id,
                    load=load,
                    truck_id=tid,
                    batch=batch,
                    wait=wait,
                    ready=ready,
                )
                if best is None or (candidate.ds, candidate.ready) < (best.ds, best.ready):
                    best = candidate

        if best is None:
            # Nothing can be placed any more: drop the rest and finish.
            for state in pending:
                for load in state.loads[state.next_index :]:
                    hints[load.lid] = {"drop": 1}
            break

        state = next(s for s in pending if s.order_id == best.order_id)
        load = best.load
        end_discharge = best.ds + load.discharge_minutes
        hints[load.lid] = {
            "start": best.batch,
            "ds": best.ds,
            "wait": best.wait,
            "late": max(0, end_discharge - load.window_end),
            "truck": best.truck_id,
            "drop": 0,
        }
        truck_free[best.truck_id] = end_discharge + load.travel_back
        slots = site_slots.setdefault(load.site_id, [])
        capacity = max(1, site_capacity.get(load.site_id, 1))
        if len(slots) < capacity:
            slots.append(end_discharge)
        else:
            slots.sort()
            slots[0] = end_discharge
        plant_free = best.batch + load.batch_minutes
        state.cursor = end_discharge
        state.next_index += 1
        if state.next_index >= len(state.loads):
            pending.remove(state)

    for load in loads:
        hints.setdefault(load.lid, {"drop": 1})
    return hints


def greedy_hints(
    req: OptimizeRequest,
    loads: list[GreedyLoad],
    candidates: dict[str, set[str]] | None = None,
) -> dict[str, dict]:
    """Return a hint per load id (drop=1 when the heuristic found no slot).

    `candidates` must mirror the model's truck compatibility pruning so the
    hint never assigns a truck the CP-SAT model has excluded.
    """
    allowed = candidates or {l.lid: {t.id for t in req.trucks if t.active} for l in loads}
    return _plan(req, loads, allowed)
