from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

TIMING_LOG = Path("output/log/pipeline_timing.json")
_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _empty_log() -> dict:
    return {"runs": [], "steps": [], "events": []}


def _read_log() -> dict:
    if not TIMING_LOG.exists():
        return _empty_log()
    try:
        with open(TIMING_LOG, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return _empty_log()

    for key in _empty_log():
        data.setdefault(key, [])
    return data


def _write_log(data: dict) -> None:
    TIMING_LOG.parent.mkdir(parents=True, exist_ok=True)
    temp_path = TIMING_LOG.with_name(
        f"{TIMING_LOG.stem}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        for attempt in range(6):
            try:
                os.replace(temp_path, TIMING_LOG)
                return
            except OSError:
                if attempt == 5:
                    return
                time.sleep(0.05 * (attempt + 1))
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def read_timing_log() -> dict:
    with _LOCK:
        return _read_log()


def reset_timing_log() -> None:
    with _LOCK:
        try:
            if TIMING_LOG.exists():
                TIMING_LOG.unlink()
        except OSError:
            pass


def record_event(label: str, event_type: str = "event", **extra) -> str:
    event_id = str(uuid.uuid4())
    now = time.time()
    with _LOCK:
        data = _read_log()
        data["events"].append(
            {
                "id": event_id,
                "label": label,
                "type": event_type,
                "timestamp": now,
                "time": _now_iso(),
                **extra,
            }
        )
        _write_log(data)
    return event_id


def start_run(label: str) -> str:
    run_id = str(uuid.uuid4())
    now = time.time()
    with _LOCK:
        data = _read_log()
        data["runs"].append(
            {
                "id": run_id,
                "label": label,
                "start": now,
                "start_time": _now_iso(),
                "end": None,
                "end_time": None,
                "duration": None,
                "status": "running",
            }
        )
        _write_log(data)
    return run_id


def finish_run(run_id: str | None, status: str = "completed", error: str | None = None) -> None:
    if not run_id:
        return
    now = time.time()
    with _LOCK:
        data = _read_log()
        for run in reversed(data["runs"]):
            if run.get("id") == run_id:
                run["end"] = now
                run["end_time"] = _now_iso()
                run["duration"] = max(0, now - run.get("start", now))
                run["status"] = status
                if error:
                    run["error"] = error
                break
        _write_log(data)


def start_step(label: str, run_id: str | None = None, category: str = "step") -> str:
    step_id = str(uuid.uuid4())
    now = time.time()
    with _LOCK:
        data = _read_log()
        data["steps"].append(
            {
                "id": step_id,
                "run_id": run_id,
                "label": label,
                "category": category,
                "start": now,
                "start_time": _now_iso(),
                "end": None,
                "end_time": None,
                "duration": None,
                "status": "running",
            }
        )
        _write_log(data)
    return step_id


def finish_step(step_id: str | None, status: str = "completed", error: str | None = None) -> None:
    if not step_id:
        return
    now = time.time()
    with _LOCK:
        data = _read_log()
        for step in reversed(data["steps"]):
            if step.get("id") == step_id:
                step["end"] = now
                step["end_time"] = _now_iso()
                step["duration"] = max(0, now - step.get("start", now))
                step["status"] = status
                if error:
                    step["error"] = error
                break
        _write_log(data)


@contextmanager
def timed_step(label: str, run_id: str | None = None, category: str = "step"):
    step_id = start_step(label, run_id=run_id, category=category)
    try:
        yield
    except Exception as exc:
        finish_step(step_id, status="error", error=str(exc))
        raise
    else:
        finish_step(step_id)
