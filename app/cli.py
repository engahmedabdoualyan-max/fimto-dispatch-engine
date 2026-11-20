"""
Command-line runner — useful for testing without the API.

    python -m app.cli samples/day_plan.json
    python -m app.cli samples/day_plan.json --json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .models import OptimizeRequest
from .solver import solve
from .validate import validate_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Solve a ready-mix dispatch plan.")
    parser.add_argument("payload", type=pathlib.Path, help="JSON request file")
    parser.add_argument("--json", action="store_true", help="print raw JSON only")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip the independent constraint check (not recommended)",
    )
    args = parser.parse_args(argv)

    req = OptimizeRequest.model_validate_json(args.payload.read_text(encoding="utf-8"))
    result = solve(req)
    report = None if args.no_validate else validate_plan(req, result)

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0 if (report is None or report.ok) else 1

    print(f"status      : {result.status} ({result.solver_status})")
    print(f"solve time  : {result.solve_ms} ms")
    print(f"objective   : {result.objective}")
    print(
        f"loads       : {result.metrics.loads_scheduled}/{result.metrics.loads_total} "
        f"(dropped {result.metrics.loads_dropped})"
    )
    print(
        f"trucks used : {result.metrics.trucks_used} · late {result.metrics.late_minutes} min · "
        f"wait {result.metrics.total_wait_minutes} min · plant {result.metrics.plant_utilization:.0%}"
    )
    print()
    header = f"{'TRUCK':<7}{'ORD':<7}{'LOAD':<5}{'M3':>5}  {'BATCH':<6}{'ARRIVE':<8}{'DISCH-END':<10}{'WAIT':>4}"
    print(header)
    print("-" * len(header))
    for a in result.assignments:
        print(
            f"{a.truck_id:<7}{a.order_id:<7}{a.load_index:<5}{a.quantity_m3:>5.1f}  "
            f"{a.batch_start:<6}{a.arrival:<8}{a.discharge_end:<10}{a.wait_minutes:>4}"
        )
    if result.unassigned:
        print("\nUNASSIGNED")
        for u in result.unassigned:
            print(f"  {u.order_id}#{u.load_index} ({u.quantity_m3} m3) — {u.reason}")

    if report is not None:
        print()
        if report.ok:
            print(f"✔ validate  : OK ({report.checked.get('assignments', 0)} loads, 0 violations)")
        else:
            print(f"✘ validate  : {len(report.violations)} violation(s)")
            for v in report.violations:
                print(f"  [{v.code}] {v.message}")
    return 0 if (report is None or report.ok) else 1


if __name__ == "__main__":
    sys.exit(main())
