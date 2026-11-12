import threading
import time
import unittest
from unittest.mock import MagicMock

from progress_report import GENERATING, LOADING_MODEL, UPLOADING, ProgressReporter

URL = "https://backend.com/api/v1/internal/tasks/task_1/progress"


class RecordingPost:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
        self.sent = threading.Event()

    def __call__(self, url, json, timeout):
        self.calls.append((url, json))
        self.sent.set()
        if self.fail:
            raise ConnectionError("backend down")

    def wait_for(self, count, timeout=2.0):
        deadline = time.monotonic() + timeout
        while len(self.calls) < count and time.monotonic() < deadline:
            time.sleep(0.005)
        return len(self.calls) >= count


class TestProgressReporter(unittest.TestCase):
    def test_without_url_nothing_is_sent(self):
        post = MagicMock()
        with ProgressReporter(None, "task_1", post=post) as progress:
            progress.start_stage(LOADING_MODEL)
            progress.musicgen_callback()(100, 100)
        post.assert_not_called()

    def test_a_stage_is_sent_right_away(self):
        post = RecordingPost()
        with ProgressReporter(URL, "task_1", heartbeat_seconds=60, post=post) as progress:
            progress.start_stage(LOADING_MODEL)
            self.assertTrue(post.wait_for(1))
        self.assertEqual(post.calls[0], (URL, {"status": "PROCESSING", "stage": "LOADING_MODEL", "progress": 10}))

    def test_heartbeat_resends_the_current_stage(self):
        post = RecordingPost()
        with ProgressReporter(URL, "task_1", heartbeat_seconds=0.02, post=post) as progress:
            progress.start_stage(GENERATING)
            self.assertTrue(post.wait_for(3))
        self.assertTrue(all(payload["stage"] == "GENERATING" for _, payload in post.calls))

    def test_nothing_is_sent_before_the_first_stage(self):
        post = RecordingPost()
        with ProgressReporter(URL, "task_1", heartbeat_seconds=0.01, post=post):
            time.sleep(0.05)
        self.assertEqual(post.calls, [])

    def test_nothing_is_sent_after_close(self):
        post = RecordingPost()
        progress = ProgressReporter(URL, "task_1", heartbeat_seconds=0.01, post=post)
        progress.start_stage(UPLOADING)
        self.assertTrue(post.wait_for(1))
        progress.close()
        sent = len(post.calls)
        progress.start_stage(UPLOADING)
        time.sleep(0.05)
        self.assertEqual(len(post.calls), sent)
        progress.close()  # idempotent

    def test_a_report_right_before_close_is_still_sent(self):
        post = RecordingPost()
        progress = ProgressReporter(URL, "task_1", heartbeat_seconds=60, post=post)
        progress.start_stage(UPLOADING)
        progress.close()
        self.assertEqual([payload["stage"] for _, payload in post.calls], ["UPLOADING"])

    def test_a_failed_request_does_not_raise(self):
        post = RecordingPost(fail=True)
        with ProgressReporter(URL, "task_1", heartbeat_seconds=60, post=post) as progress:
            progress.start_stage(LOADING_MODEL)
            self.assertTrue(post.wait_for(1))
            progress.start_stage(GENERATING)
            self.assertTrue(post.wait_for(2))

    def test_musicgen_progress_is_mapped_and_throttled(self):
        progress = ProgressReporter(None, "task_1")
        progress.report = MagicMock()
        callback = progress.musicgen_callback(start=30, end=85)
        for generated in range(0, 1501, 15):
            callback(generated, 1500)
        values = [c.args[1] for c in progress.report.call_args_list]
        self.assertEqual(values[-1], 85)
        self.assertTrue(all(b - a >= 5 for a, b in zip(values, values[1:])))
        self.assertTrue(all(c.args[0] == "GENERATING" for c in progress.report.call_args_list))
        callback(0, 0)  # no division by zero


if __name__ == '__main__':
    unittest.main()
