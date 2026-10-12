"""Tempos opt-in do pipeline. Sem arquivo ou relógio no caminho desabilitado."""

from collections import Counter, defaultdict
from contextvars import ContextVar
from functools import wraps
import json
import math
import statistics
import sys
import time


_active = ContextVar("pae_diagnostics", default=None)


def timed_call(stage, function, *args, **kwargs):
    profiler = _active.get()
    if profiler is None or profiler.record is None:
        return function(*args, **kwargs)
    start = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        profiler.record["stages_s"][stage] += time.perf_counter() - start


def timed(stage):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            return timed_call(stage, function, *args, **kwargs)
        return wrapped
    return decorate


def note(**values):
    profiler = _active.get()
    if profiler is not None and profiler.record is not None:
        profiler.record.update(values)


def milestone(name):
    profiler = _active.get()
    if profiler is not None and profiler.record is not None:
        profiler.record[name] = time.perf_counter() - profiler.started


def start_cycle():
    profiler = _active.get()
    if profiler is not None:
        profiler.start()


def finish_cycle(outcome):
    profiler = _active.get()
    if profiler is not None:
        profiler.finish(outcome)


class PipelineProfiler:
    def __init__(self, stream):
        self.stream = stream
        self.record = None
        self.started = None
        self.sequence = 0
        self.samples = defaultdict(list)
        self.outcomes = Counter()
        self.wait_started = None
        self.turn_ready = False

    def start(self):
        now = time.perf_counter()
        self.sequence += 1
        self.record = dict(type="cycle", cycle=self.sequence, stages_s=defaultdict(float))
        if self.started is not None:
            self.record["capture_interval_s"] = now - self.started
        self.started = now

    def finish(self, outcome):
        if self.record is None:
            return
        if outcome == "interrupted" and not self.record.get("capture_complete"):
            self.record = None
            return
        row = self.record
        row.update(outcome=outcome, processing_s=time.perf_counter() - self.started)
        stages = row["stages_s"]
        if "hero_total" in stages:
            stages["hero_location"] = max(0, stages["hero_total"] - stages.get("hero_classification", 0))
        if row.get("hero_turn"):
            if self.wait_started is None and not self.turn_ready:
                self.wait_started = self.started
            if row.get("recommendation_available"):
                row["recommendation_latency_s"] = row.get("recommendation_emitted_s", row["processing_s"])
                if self.wait_started is not None:
                    row["observed_wait_to_recommendation_s"] = (
                        self.started + row["recommendation_latency_s"] - self.wait_started)
                    self.wait_started = None
                self.turn_ready = True
        else:
            self.wait_started = None
            self.turn_ready = False
        # O tempo de espera começa na primeira captura ativa observada, não no
        # instante (desconhecido) em que o servidor disponibilizou os botões.
        for name, value in stages.items():
            self.samples[name].append(value)
        for name in ("processing_s", "capture_interval_s", "recommendation_latency_s",
                     "observed_wait_to_recommendation_s"):
            if name in row:
                self.samples[name].append(row[name])
        self.outcomes[outcome] += 1
        self.stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.stream.flush()
        self.record = None

    def summary(self):
        def stats(values):
            ordered = sorted(values)
            return dict(n=len(values), median=statistics.median(values),
                        p95=ordered[math.ceil(.95 * len(values)) - 1], maximum=ordered[-1])
        return dict(type="summary", units="seconds", outcomes=dict(self.outcomes),
                    metrics={name: stats(values) for name, values in self.samples.items()})


class _TimedOutput:
    def __init__(self, output):
        self.output = output

    def __getattr__(self, name):
        return getattr(self.output, name)

    def write(self, text):
        return timed_call("terminal", self.output.write, text)

    def flush(self):
        return timed_call("terminal", self.output.flush)


def profile_session(function):
    @wraps(function)
    def wrapped(*args, profile_jsonl=None, **kwargs):
        if profile_jsonl is None:
            return function(*args, **kwargs)
        with open(profile_jsonl, "a", encoding="utf-8") as stream:
            profiler = PipelineProfiler(stream)
            token = _active.set(profiler)
            output = sys.stdout
            sys.stdout = _TimedOutput(output)
            try:
                return function(*args, **kwargs)
            finally:
                sys.stdout = output
                try:
                    profiler.finish("interrupted")
                    stream.write(json.dumps(profiler.summary(), ensure_ascii=False) + "\n")
                finally:
                    _active.reset(token)
    return wrapped
