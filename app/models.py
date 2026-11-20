"""
Request / response contracts for the dispatch optimizer.

Units
-----
* volume      : cubic metres (m3)
* time        : integer minutes from midnight (e.g. 08:30 -> 510)
* rate        : m3 per hour
* durations   : integer minutes

The contract is deliberately database-agnostic: the Next.js API maps
`orders` / `fleet_vehicles` / `delivery_sites` / `batch_plants` rows into
this payload (see `src/app/api/dispatch/optimize/route.ts`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Truck(BaseModel):
    """A mixer truck (drum capacity + availability window)."""

    id: str
    code: str | None = Field(default=None, description="Vehicle code, e.g. T01")
    capacity_m3: float = Field(gt=0, le=30)
    available_from_minute: int = Field(default=0, ge=0, le=1439)
    available_to_minute: int = Field(default=1440, ge=1, le=2880)
    driver_id: str | None = None
    active: bool = True

    @model_validator(mode="after")
    def _check_window(self) -> "Truck":
        if self.available_to_minute <= self.available_from_minute:
            raise ValueError(f"truck {self.id}: available_to must be after available_from")
        return self


class Site(BaseModel):
    """Delivery site (customer location)."""

    id: str
    name: str | None = None
    travel_from_plant_min: int = Field(ge=1, le=600)
    travel_to_plant_min: int | None = Field(default=None, ge=1, le=600)
    discharge_rate_m3_per_hour: float = Field(default=20, gt=0, le=200)
    max_trucks_on_site: int = Field(default=2, ge=1, le=20)

    @property
    def return_minutes(self) -> int:
        return self.travel_to_plant_min or self.travel_from_plant_min


class Order(BaseModel):
    """A pouring order that must be split into truck loads."""

    id: str
    site_id: str
    volume_m3: float = Field(gt=0, le=2000)
    pour_rate_m3_per_hour: float | None = Field(default=None, gt=0, le=200)
    window_start_minute: int = Field(ge=0, le=2880)
    window_end_minute: int = Field(ge=1, le=2880)
    max_trucks_on_site: int | None = Field(default=None, ge=1, le=20)
    priority: int = Field(default=0, ge=0, le=9)

    @model_validator(mode="after")
    def _check_window(self) -> "Order":
        if self.window_end_minute <= self.window_start_minute:
            raise ValueError(f"order {self.id}: window_end must be after window_start")
        return self


class Plant(BaseModel):
    """Batching plant."""

    id: str
    code: str | None = None
    rate_m3_per_hour: float = Field(gt=0, le=400)
    load_time_minutes: int = Field(default=5, ge=0, le=60)


class Weights(BaseModel):
    """Objective weights (integers — CP-SAT requires an integer objective)."""

    drop: int = Field(
        default=1_000_000,
        ge=1,
        description="cost of not scheduling a load (must exceed late*max_late)",
    )
    late: int = Field(default=200, ge=1, description="cost per minute after the window")
    makespan: int = Field(default=1, ge=0, description="cost per minute of the whole day")
    truck: int = Field(default=500, ge=0, description="cost of using one extra truck")
    wait: int = Field(default=1, ge=0, description="cost per minute a truck waits on site")
    gap: int = Field(default=50, ge=0, description="cost per minute of idle pour time between loads")


class Options(BaseModel):
    max_drum_minutes: int = Field(
        default=90, ge=30, le=240, description="max minutes from batching to end of discharge"
    )
    max_late_minutes: int = Field(default=240, ge=0, le=720)
    max_gap_minutes: int | None = Field(
        default=30,
        ge=0,
        le=180,
        description="max gap between consecutive discharges of the same order (None = off)",
    )
    time_limit_seconds: float = Field(default=10, gt=0, le=120)
    workers: int = Field(default=8, ge=1, le=16)
    seed: int = 42
    reference_load_m3: float | None = Field(
        default=None,
        gt=0,
        description="fixed load size; defaults to the fleet's most common capacity",
    )
    weights: Weights = Field(default_factory=Weights)


class OptimizeRequest(BaseModel):
    plan_date: str | None = None
    horizon_start_minute: int = Field(default=300, ge=0, le=1439)
    horizon_end_minute: int = Field(default=1320, ge=60, le=2880)
    plant: Plant
    trucks: list[Truck]
    orders: list[Order]
    sites: list[Site]
    options: Options = Field(default_factory=Options)


class Assignment(BaseModel):
    order_id: str
    load_index: int
    truck_id: str
    truck_code: str | None = None
    quantity_m3: float
    batch_start_minute: int
    arrival_minute: int
    discharge_start_minute: int
    discharge_end_minute: int
    return_to_plant_minute: int
    wait_minutes: int
    sequence: int = 1

    # convenience HH:MM strings for the UI
    batch_start: str = ""
    arrival: str = ""
    discharge_end: str = ""


class Unassigned(BaseModel):
    order_id: str
    load_index: int
    quantity_m3: float
    reason: Literal[
        "NO_FEASIBLE_TRUCK",
        "TIME_WINDOW",
        "CAPACITY_OR_CONGESTION",
        "FRESHNESS_LIMIT",
    ]


class Metrics(BaseModel):
    loads_total: int = 0
    loads_scheduled: int = 0
    loads_dropped: int = 0
    trucks_used: int = 0
    late_minutes: int = 0
    orders_late: int = 0
    total_wait_minutes: int = 0
    avg_wait_minutes: float = 0.0
    makespan_minutes: int = 0
    plant_utilization: float = 0.0


class OptimizeResponse(BaseModel):
    status: Literal["OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN", "MODEL_INVALID", "ERROR"]
    solver_status: str
    objective: int | None = None
    solve_ms: int = 0
    plan_date: str | None = None
    assignments: list[Assignment] = Field(default_factory=list)
    unassigned: list[Unassigned] = Field(default_factory=list)
    trucks: list[dict] = Field(default_factory=list)
    metrics: Metrics = Field(default_factory=Metrics)
    message: str | None = None


def minute_to_hhmm(minute: int) -> str:
    """510 -> '08:30'. Handles minutes past midnight."""
    minute = int(minute) % (24 * 60)
    return f"{minute // 60:02d}:{minute % 60:02d}"
