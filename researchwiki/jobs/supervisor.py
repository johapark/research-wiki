"""Caller-owned local worker processes; no daemon or implicit paid retry."""

from dataclasses import dataclass
import subprocess
import sys

from filelock import Timeout

from .environment import child_environment
from .events import pending_relays, read_events
from .files import JobConflict, JobStoreUnavailable
from .store import JobStore


@dataclass
class WorkerHandle:
    job_id: str
    process: subprocess.Popen


class JobSupervisor:
    def __init__(self, library_root, *, store_root=None, environment=None):
        self.store = JobStore(library_root, store_root=store_root)
        self.environment = environment

    def submit_ingest(self, request):
        return self.store.submit_ingest(request)

    def get_job(self, job_id):
        return self.store.get_job(job_id)

    def list_jobs(self):
        return self.store.list_jobs()

    def read_events(self, job_id, *, after_sequence=0):
        return read_events(self.store, job_id, after_sequence=after_sequence)

    def pending_relays(self, job_id):
        return pending_relays(self.store, job_id)

    def cancel_queued_job(self, job_id):
        return self.store.cancel_queued_job(job_id)

    def retry_job(self, job_id):
        return self.store.retry_job(job_id)

    def reconcile_job(self, job_id, *, timeout=None):
        """Repair in a fresh library/profile process, without repeating model work."""
        self.store.get_job(job_id)
        command = [sys.executable, "-m", "researchwiki.jobs.recovery", "--library-root",
                   str(self.store.library_root), "--job-id", job_id]
        if self.store.base is not None:
            command.extend(["--store-root", str(self.store.base)])
        completed = subprocess.run(command, cwd=self.store.library_root,
                                   env=child_environment(self.environment), stdin=subprocess.DEVNULL,
                                   timeout=timeout)
        if completed.returncode == 1:
            raise JobConflict("job has a live owner; reconciliation refused")
        if completed.returncode:
            raise JobStoreUnavailable("isolated job reconciliation failed")
        return self.store.get_job(job_id)

    def start_job(self, job_id, *, stdout=None, stderr=None):
        record = self.store.get_job(job_id)
        if record.execution != "queued":
            raise JobConflict("only queued jobs can start")
        command = [sys.executable, "-m", "researchwiki.jobs.worker", "--library-root",
                   str(self.store.library_root), "--job-id", job_id]
        if self.store.base is not None:
            command.extend(["--store-root", str(self.store.base)])
        process = subprocess.Popen(
            command, cwd=self.store.library_root, env=child_environment(self.environment),
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
        )
        return WorkerHandle(job_id, process)

    def wait(self, handle, *, timeout=None):
        handle.process.wait(timeout=timeout)
        return self.refresh_job(handle.job_id)

    def refresh_job(self, job_id):
        """Mark owner loss conservatively; explicit reconcile_job resolves evidence.

        Acquiring the same OS lock as the worker prevents a liveness guess
        based on a reused PID. A live worker is never reconciled underneath.
        """
        try:
            with self.store.owner_lock(job_id):
                return self.store.mark_interrupted(job_id)
        except Timeout:
            return self.store.get_job(job_id)
