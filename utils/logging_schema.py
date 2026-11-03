"""Structured JSON logging schema for EP²-MPC trial/run events."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Tuple

REQUIRED_TRIAL_FIELDS = (
    "event_type",
    "method",
    "scenario",
    "seed",
    "trial_index",
    "solver_status",
    "timeout",
    "collision",
    "success",
    "energy",
    "time",
    "solver_ms",
)

REQUIRED_RUN_FIELDS = (
    "event_type",
    "method",
    "scenario",
    "n_trials",
    "success_rate",
    "solver_status_counts",
)


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def make_trial_event(
    *,
    method: str,
    scenario: str,
    seed: int,
    trial_index: int,
    solver_status: str,
    timeout: bool,
    collision: bool,
    success: bool,
    energy: float,
    time: float,
    solver_ms: float,
) -> dict:
    return {
        "event_type": "trial",
        "method": method,
        "scenario": scenario,
        "seed": int(seed),
        "trial_index": int(trial_index),
        "solver_status": str(solver_status),
        "timeout": bool(timeout),
        "collision": bool(collision),
        "success": bool(success),
        "energy": float(energy),
        "time": float(time),
        "solver_ms": float(solver_ms),
    }


def make_run_event(
    *,
    method: str,
    scenario: str,
    n_trials: int,
    success_rate: float,
    solver_status_counts: Dict[str, int],
    extra: Dict[str, Any] | None = None,
) -> dict:
    payload = {
        "event_type": "run",
        "method": method,
        "scenario": scenario,
        "n_trials": int(n_trials),
        "success_rate": float(success_rate),
        "solver_status_counts": {str(k): int(v) for k, v in solver_status_counts.items()},
    }
    if extra:
        payload.update(_to_jsonable(extra))
    return payload


def validate_event(event: dict) -> Tuple[bool, List[str]]:
    if not isinstance(event, dict):
        return False, ["event must be object"]
    e_type = event.get("event_type")
    req = REQUIRED_TRIAL_FIELDS if e_type == "trial" else REQUIRED_RUN_FIELDS
    errors = [f"missing:{k}" for k in req if k not in event]
    if e_type not in ("trial", "run"):
        errors.append("invalid:event_type")
    return len(errors) == 0, errors


def validate_log_file(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {"pass": False, "errors": [f"missing:{path}"], "count": 0}
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        return {"pass": False, "errors": [f"parse_error:{e}"], "count": 0}
    if not isinstance(payload, list):
        return {"pass": False, "errors": ["root_not_list"], "count": 0}
    errors: List[str] = []
    for i, ev in enumerate(payload):
        ok, errs = validate_event(ev)
        if not ok:
            errors.extend([f"{i}:{e}" for e in errs])
    return {"pass": len(errors) == 0, "errors": errors, "count": len(payload)}


def save_events(path: str, events: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(events), f, indent=2, default=float)
