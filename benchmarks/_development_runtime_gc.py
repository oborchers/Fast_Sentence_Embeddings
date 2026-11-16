"""Fixed GC counters independent of the event recorder and its reentrancy guard."""

from __future__ import annotations

import threading
import time
from array import array

SLOTS = 64


class Collector:
    def __init__(self):
        self.threads = array("q", [0]) * SLOTS
        self.start_wall = array("q", [0]) * SLOTS
        self.start_cpu = array("q", [0]) * SLOTS
        self.wall = array("q", [0]) * SLOTS
        self.cpu = array("q", [0]) * SLOTS
        self.count = array("q", [0]) * SLOTS
        self.losses = 0

    def _slot(self):
        thread = threading.get_ident()
        slot = 0
        while slot < SLOTS:
            if self.threads[slot] == thread:
                return slot
            if self.threads[slot] == 0:
                self.threads[slot] = thread
                return slot
            slot += 1
        self.losses += 1
        return -1

    def callback(self, phase, _info):
        slot = self._slot()
        if slot < 0:
            return
        if phase == "start":
            self.start_wall[slot] = time.perf_counter_ns()
            self.start_cpu[slot] = time.thread_time_ns()
        elif self.start_wall[slot]:
            self.wall[slot] += time.perf_counter_ns() - self.start_wall[slot]
            self.cpu[slot] += time.thread_time_ns() - self.start_cpu[slot]
            self.count[slot] += 1
            self.start_wall[slot] = 0
        else:
            self.losses += 1

    def export(self):
        return {
            "threads": {
                str(thread): {
                    "wall_seconds": self.wall[slot] / 1e9,
                    "thread_cpu_seconds": self.cpu[slot] / 1e9,
                    "collections": self.count[slot],
                }
                for slot, thread in enumerate(self.threads)
                if thread
            },
            "callback_losses": self.losses + sum(value != 0 for value in self.start_wall),
            "loss_policy": "warning only; GC coverage does not invalidate worker events",
            "scope": "callbacks allocate no GC-tracked containers; their own overhead remains",
        }
