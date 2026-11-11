"""Background execution of the REAL Gemini optimisation agent (no Streamlit imports).

The agent runs in a worker thread with its own asyncio loop, so the dashboard stays responsive. Progress lines are
collected (sanitised) and every tool call is also persisted by the agent itself, which is what the live trace view reads.
There is deliberately NO scripted/mock fallback in this module: without credentials start() raises MissingCredentialsError.
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from agent.payments.models import AgentConfig, MissingCredentialsError
from dashboard.payments import paths
from dashboard.payments.errors import ErrorInfo, classify_error
from dashboard.payments.redaction import sanitize

MAX_PROGRESS_LINES = 600


class JobBusy(RuntimeError):
    """Another optimisation is already running."""


@dataclass
class Job:
    job_id: str
    objective_text: str
    overrides: dict[str, Any]
    config: dict[str, Any]  # AgentConfig.redacted(): never contains the API key
    state: str = "RUNNING"  # RUNNING | COMPLETED | COMPLETED_WITH_LLM_ERROR | FAILED
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = None
    progress: list[str] = field(default_factory=list)
    report: Optional[dict] = None
    error: Optional[ErrorInfo] = None
    execution_id: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def log(self, msg: str) -> None:
        with self._lock:
            self.progress.append(sanitize(msg, max_str=700))
            del self.progress[:-MAX_PROGRESS_LINES]

    def lines(self) -> list[str]:
        with self._lock:
            return list(self.progress)

    @property
    def running(self) -> bool:
        return self.state == "RUNNING"


def _default_agent_factory(cfg: AgentConfig, progress: Callable[[str], None]):
    from agent.payments.agent import PaymentAgent  # real GeminiBackend; raises MissingCredentialsError without a key
    return PaymentAgent(cfg, progress=progress)


class JobManager:
    def __init__(self, agent_factory: Optional[Callable[[AgentConfig, Callable[[str], None]], Any]] = None):
        self._factory = agent_factory or _default_agent_factory
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    def active(self) -> Optional[Job]:
        with self._lock:
            return next((j for j in self._jobs.values() if j.running), None)

    def latest(self) -> Optional[Job]:
        with self._lock:
            return max(self._jobs.values(), key=lambda j: j.started_at, default=None)

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def start(self, objective_text: str, overrides: dict[str, Any], *, max_iterations: int = 8, exploration_trials: int = 10,
              final_trials: int = 20, model: Optional[str] = None) -> Job:
        cfg_kw: dict[str, Any] = dict(max_iterations=max_iterations, exploration_trials=exploration_trials, final_trials=final_trials,
                                      db_path=paths.sim_db_path(), agent_db_path=paths.agent_db_path())
        if model:
            cfg_kw["model"] = model
        cfg = AgentConfig.from_env(**cfg_kw)  # raises MissingCredentialsError synchronously: never starts without a real key
        with self._lock:
            if any(j.running for j in self._jobs.values()):
                raise JobBusy("An optimisation is already running; wait for it to finish.")
            job = Job(uuid.uuid4().hex[:12], objective_text, dict(overrides), cfg.redacted())
            self._jobs[job.job_id] = job
        t = threading.Thread(target=self._run, args=(job, cfg), name=f"agent-job-{job.job_id}", daemon=True)
        self._threads[job.job_id] = t
        t.start()
        return job

    def wait(self, job_id: str, timeout: float = 60.0) -> Job:
        """Test/CLI helper: block until the job finishes."""
        t = self._threads.get(job_id)
        if t:
            t.join(timeout)
        return self._jobs[job_id]

    def _run(self, job: Job, cfg: AgentConfig) -> None:
        try:
            agent = self._factory(cfg, job.log)
            report = asyncio.run(agent.run(job.objective_text, **job.overrides))
            job.report, job.execution_id = report.to_dict(), report.execution_id
            if report.termination_reason == "llm_error":
                note = next((n for n in report.notes if "LLM error" in n), "LLM error")
                job.error, job.state = classify_error(note), "COMPLETED_WITH_LLM_ERROR"
            else:
                job.state = "COMPLETED"
        except BaseException as e:  # noqa: BLE001 -- surfaced to the UI, never swallowed
            job.error, job.state = classify_error(e), "FAILED"
            job.log(f"FAILED: {type(e).__name__}")
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat()


_MANAGER: Optional[JobManager] = None
_MANAGER_LOCK = threading.Lock()


def get_manager() -> JobManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = JobManager()
        return _MANAGER


def set_manager(m: Optional[JobManager]) -> None:
    """Replace the process-wide manager (tests)."""
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = m
