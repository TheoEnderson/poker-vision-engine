"""Medição opt-in sem suprimir validação, decisões ou leituras de cada frame."""
import io
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch

import main
from src import diagnostics
from src.ocr.ocr_reader import parse_action_button_text
from tests import test_hand_cycle_integration as cycle
from tests.test_same_street_decisions import FLOP, TURN, RIVER


class TestDiagnostics(unittest.TestCase):
    def test_disabled_session_does_not_open_file_or_read_clock(self):
        @diagnostics.profile_session
        def work():
            diagnostics.start_cycle()
            result = diagnostics.timed_call("stage", lambda: 42)
            diagnostics.note(value=result)
            diagnostics.finish_cycle("ok")
            return result
        with patch("builtins.open") as opened, patch.object(diagnostics.time, "perf_counter") as clock:
            self.assertEqual(work(), 42)
            opened.assert_not_called()
            clock.assert_not_called()

    def test_processing_interval_and_observed_wait_have_different_meanings(self):
        stream = io.StringIO()
        profiler = diagnostics.PipelineProfiler(stream)
        now = [100.0]
        token = diagnostics._active.set(profiler)
        try:
            with patch.object(diagnostics.time, "perf_counter", side_effect=lambda: now[0]):
                profiler.start()
                diagnostics.note(capture_complete=True, hero_turn=True)
                now[0] += 2
                profiler.finish("recognition_unconfirmed")
                now[0] += .5
                profiler.start()
                diagnostics.note(capture_complete=True, hero_turn=True, recommendation_available=True)
                now[0] += 1
                profiler.finish("recommendation")
        finally:
            diagnostics._active.reset(token)
        first, second = map(json.loads, stream.getvalue().splitlines())
        self.assertEqual(first["processing_s"], 2)
        self.assertEqual(second["processing_s"], 1)
        self.assertEqual(second["capture_interval_s"], 2.5)
        self.assertEqual(second["recommendation_latency_s"], 1)
        self.assertEqual(second["observed_wait_to_recommendation_s"], 3.5)
        self.assertEqual(profiler.summary()["metrics"]["processing_s"], dict(n=2, median=1.5, p95=2, maximum=2))


class TestProfiledLive(unittest.TestCase):
    setUp = cycle.TestHandCycleIntegration.setUp

    def live(self, frames):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latency.jsonl"
            with patch.object(cycle, "run_live", side_effect=lambda **kw: main.run_live(profile_jsonl=path, **kw)):
                output = cycle.TestHandCycleIntegration.live(self, frames)
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.rows = [r for r in records if r["type"] == "cycle"]
        self.summary = records[-1]
        return output

    def test_board_detected_while_hero_unreadable_then_recovers_without_reset(self):
        self.live([cycle.capture(), cycle.capture(hero=[], board=FLOP),
                   cycle.capture(board=FLOP), cycle.capture(board=TURN), cycle.capture(board=RIVER)])
        self.assertEqual([r["state"] for r in self.rows], ["PRE_FLOP", "PRE_FLOP", "FLOP", "TURN", "RIVER"])
        self.assertEqual(self.rows[1]["board_observed"], FLOP)
        self.assertEqual(self.rows[1]["outcome"], "recognition_unconfirmed")
        self.assertLess(self.rows[1]["board_detected_s"], self.rows[1]["hand_observed_s"])
        self.assertEqual(self.reset_count, 0)
        self.assertEqual(self.calculate_equity.call_count, 4)

    def test_all_readers_still_run_on_each_capture_and_caches_are_preserved(self):
        self.live([cycle.capture(), cycle.capture(), cycle.capture(pot=25, call=10)])
        for row in self.rows:
            for stage in ("capture", "hero_total", "board", "pot_ocr", "hand_ids_ocr", "buttons_ocr", "dealer", "stack_ocr", "fsm", "financial_validation", "terminal"):
                self.assertIn(stage, row["stages_s"])
        self.assertEqual(self.calculate_equity.call_count, 1)
        self.assertEqual(self.risk.call_count, 2)
        self.assertIs(self.analyses[0], self.analyses[1])
        self.assertEqual(self.summary["metrics"]["monte_carlo"]["n"], 1)
        self.assertEqual(self.summary["metrics"]["strategy"]["n"], 2)

    def test_parallel_readers_finish_same_frame_before_fsm_observation(self):
        frames = [cycle.capture(), cycle.capture(pot=25, call=10)]
        originals = list(frames)
        before = deepcopy(originals)
        completed = []
        barrier = Barrier(4)
        observe = self.app.observe_hand
        observations = []

        class SynchronizedPool(ThreadPoolExecutor):
            def submit(pool, fn, *args, **kwargs):
                # copy_context().run(timed_call, stage, reader, frame, ...)
                _, stage, _, frame = args

                def work():
                    barrier.wait(timeout=5)  # Guarda contra deadlock, não limiar de desempenho.
                    result = fn(*args, **kwargs)
                    completed.append((stage, id(frame)))
                    return result
                return super().submit(work)

        def checked_observation(*args, **kwargs):
            index = len(observations)
            self.assertEqual(len(completed), 4 * (index + 1))
            self.assertEqual(set(completed[-4:]), {
                (stage, id(originals[index]))
                for stage in ("pot_ocr", "hand_ids_ocr", "buttons_ocr", "dealer")
            })
            observations.append(index)
            return observe(*args, **kwargs)

        with patch("main.ThreadPoolExecutor", SynchronizedPool), \
             patch.object(self.app, "observe_hand", side_effect=checked_observation):
            self.live(frames)
        self.assertEqual(observations, [0, 1])
        self.assertEqual(originals, before)

    def test_out_of_turn_and_financial_suspension_are_distinct(self):
        self.live([cycle.capture(turn=False), cycle.capture(pot=None), cycle.capture()])
        self.assertEqual([r["outcome"] for r in self.rows], ["out_of_turn", "financial_suspension", "recommendation"])
        self.assertEqual(self.calculate_equity.call_count, 1)
        self.assertNotIn("monte_carlo", self.rows[1]["stages_s"])

    def test_live_all_in_keeps_financial_reason_and_blocks_math(self):
        from tests import test_hand_cycle_integration as harness
        allin = parse_action_button_text("FOLD ALL-IN 1195")
        with patch.object(harness, "context", return_value=allin):
            output = self.live([cycle.capture(pot=2039, stack=1190, call=None)])
        self.assertEqual(self.rows[0]["outcome"], "financial_suspension")
        self.assertIn("pote elegível", output)
        self.calculate_equity.assert_not_called()
        self.risk.assert_not_called()


if __name__ == "__main__":
    unittest.main()
