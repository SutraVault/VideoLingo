"""
Background task runner for Streamlit with pause/resume/stop control.

Usage:
    runner = TaskRunner.get(st.session_state)
    runner.start(steps)  # list of (label, callable) tuples
    runner.pause() / runner.resume() / runner.stop()
    runner.state  # "idle" | "running" | "paused" | "stopped" | "completed" | "error"
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from core.utils.timing import finish_run, finish_step, start_run, start_step


class StopTask(Exception):
    """Raised when the task is stopped by user."""

    pass


@dataclass
class TaskRunner:
    """Manages a background thread that executes a sequence of steps with pause/stop control."""

    # Public read-only state
    state: str = "idle"  # idle | running | paused | stopped | completed | error
    current_step: int = -1  # 0-indexed, -1 = not started
    total_steps: int = 0
    current_label: str = ""
    error_msg: str = ""

    # Internal
    _pause_event: threading.Event = field(default_factory=threading.Event)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _steps: list = field(default_factory=list)
    _run_id: str | None = None

    def __post_init__(self):
        self._pause_event.set()  # not paused initially

    # ------ Singleton per session_state ------
    @staticmethod
    def get(session_state, key: str = "_task_runner") -> "TaskRunner":
        """Get or create a TaskRunner stored in Streamlit session_state."""
        if key not in session_state:
            session_state[key] = TaskRunner()
        return session_state[key]

    # ------ Control API ------

    def start(self, steps: list[tuple[str, Callable]], run_label: str | None = None):
        """Start executing steps in a background thread.

        Args:
            steps: list of (label, callable) — each callable takes no args.
        """
        if self.state == "running" or self.state == "paused":
            return  # already running

        self._steps = steps
        self.total_steps = len(steps)
        self.current_step = -1
        self.current_label = ""
        self.error_msg = ""
        self.state = "running"
        self._run_id = start_run(run_label or "Task")

        self._pause_event.set()
        self._stop_event.clear()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self):
        if self.state == "running":
            self._pause_event.clear()
            self.state = "paused"

    def resume(self):
        if self.state == "paused":
            self._pause_event.set()
            self.state = "running"

    def stop(self):
        """Request stop. The task will halt before the next step."""
        if self.state in ("running", "paused"):
            self._stop_event.set()
            self._pause_event.set()  # unblock if paused so thread can exit
            self.state = "stopped"
            finish_run(self._run_id, status="stopped")

    def reset(self):
        """Reset to idle state (only when not running)."""
        if self.state not in ("running", "paused"):
            self.state = "idle"
            self.current_step = -1
            self.total_steps = 0
            self.current_label = ""
            self.error_msg = ""
            self._steps = []
            self._run_id = None

    def retry_from_failed_step(self):
        """Retry the failed step and all later steps without repeating completed work."""
        if self.state != "error" or not self._steps:
            return
        failed_index = max(0, self.current_step)
        remaining_steps = self._steps[failed_index:]
        self.start(remaining_steps, run_label="Retry failed task")

    @property
    def is_active(self) -> bool:
        return self.state in ("running", "paused")

    @property
    def is_done(self) -> bool:
        return self.state in ("completed", "stopped", "error")

    @property
    def progress(self) -> float:
        """0.0 to 1.0"""
        if self.total_steps == 0:
            return 0.0
        return min((self.current_step + 1) / self.total_steps, 1.0)

    # ------ Internal ------

    def _run(self):
        """Execute steps sequentially in background thread."""
        try:
            for i, (label, func) in enumerate(self._steps):
                # Check stop before each step
                if self._stop_event.is_set():
                    self.state = "stopped"
                    return

                # Block if paused
                self._pause_event.wait()

                # Check stop again after resume
                if self._stop_event.is_set():
                    self.state = "stopped"
                    return

                self.current_step = i
                self.current_label = label
                step_id = start_step(label, run_id=self._run_id)
                try:
                    func()
                except Exception as e:
                    finish_step(step_id, status="error", error=str(e))
                    raise
                else:
                    finish_step(step_id)

            self.state = "completed"
            finish_run(self._run_id)
        except Exception as e:
            self.error_msg = str(e)
            self.state = "error"
            finish_run(self._run_id, status="error", error=str(e))
            log_dir = Path("output/log")
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / "task_runner_error.log", "a", encoding="utf-8") as f:
                f.write(f"\n--- Error in step: {self.current_label} ---\n")
                f.write(traceback.format_exc())
