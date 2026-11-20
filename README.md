# Fimto dispatch optimizer (OR-Tools CP-SAT)

FastAPI service that solves the concrete-delivery dispatch problem
(vehicles × orders × time windows, mixer capacity, travel time).

The ERP API calls `POST /optimize`; if this service is unreachable the API
falls back to its built-in greedy heuristic, so dispatch never blocks.

`GET /health` is used as the Render health check.
