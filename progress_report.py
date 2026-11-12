"""Report a running job's progress to the backend (progress_url), stage by stage.

The backend relays each report to the client as an SSE progress event and extends its wait, so a
long job (AMT extension, MusicGen) is not cut off by the backend's fixed timeout.
- A background thread sends every report, and re-sends the latest one every HEARTBEAT_SECONDS
  while a long step runs, so the backend hears from the job even when no stage changes.
- Reporting is best-effort: without a progress_url nothing is sent, and a failed request is only
  logged. It never fails the job.
- close() stops reporting before the completion callback, so no progress arrives after it
  (a report not sent yet is sent first).

Payload: {"status": "PROCESSING", "stage": <stage>, "progress": <0~90>}
"""
import threading

import requests

HEARTBEAT_SECONDS = 30
REQUEST_TIMEOUT_SECONDS = 5
# MusicGen token progress is reported only when it moves this many points
MIN_PROGRESS_STEP = 5

# Stage -> progress when the stage starts. 100 is left to the completion callback.
LOADING_MODEL = ('LOADING_MODEL', 10)
EXTENDING_MELODY = ('EXTENDING_MELODY', 20)
GENERATING = ('GENERATING', 30)
GENERATING_END = 85
UPLOADING = ('UPLOADING', 90)


class ProgressReporter:
    def __init__(self, progress_url, task_id, heartbeat_seconds=HEARTBEAT_SECONDS, post=None):
        self.progress_url = progress_url
        self.task_id = task_id
        self.heartbeat_seconds = heartbeat_seconds
        self._post = post
        self._state = None
        self._pending = False
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._thread = None
        if progress_url:
            self._thread = threading.Thread(target=self._run, name=f"progress-{task_id}", daemon=True)
            self._thread.start()

    def report(self, stage, progress):
        """Set the current stage and send it now (from the background thread)."""
        if self._thread is None or self._closed.is_set():
            return
        with self._lock:
            self._state = (stage, progress)
            self._pending = True
        self._wake.set()

    def start_stage(self, stage):
        self.report(*stage)

    def musicgen_callback(self, start=GENERATING[1], end=GENERATING_END):
        """MusicGen progress callback (generated_tokens, tokens_to_generate) mapped onto start~end."""
        last = [start]

        def callback(generated_tokens, tokens_to_generate):
            if tokens_to_generate <= 0:
                return
            fraction = min(generated_tokens / tokens_to_generate, 1.0)
            progress = round(start + (end - start) * fraction)
            if progress - last[0] >= MIN_PROGRESS_STEP:
                last[0] = progress
                self.report(GENERATING[0], progress)

        return callback

    def close(self):
        """Stop reporting; waits for a report in flight so it arrives before the completion callback."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=REQUEST_TIMEOUT_SECONDS + 1)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _run(self):
        while True:
            self._wake.wait(self.heartbeat_seconds)
            self._wake.clear()
            with self._lock:
                state, pending = self._state, self._pending
                self._pending = False
            if self._closed.is_set():
                # A report made just before close() is still sent, so the last stage is not lost
                if pending:
                    self._send(*state)
                return
            if state is not None:
                self._send(*state)

    def _send(self, stage, progress):
        payload = {"status": "PROCESSING", "stage": stage, "progress": progress}
        try:
            (self._post or requests.post)(self.progress_url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        except Exception as e:
            print(f"[{self.task_id}] Progress report failed ({stage} {progress}): {e}")
