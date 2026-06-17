from __future__ import annotations

import queue
import sys
import threading
from typing import Optional

import torch

from .source import RGBDSource, RGBDFrame


class AsyncRGBDSource(RGBDSource):

    def __init__(
        self,
        source: RGBDSource,
        num_workers: int = 4,
        queue_size: int = 8,
        pin_memory: bool = True,
    ):
        self._source = source
        self._num_workers = num_workers
        self._queue_size = queue_size
        self._pin_memory = pin_memory and torch.cuda.is_available()

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []
        self._started = False

        self._next_idx = 0
        self._total_frames = len(source)
        self._idx_lock = threading.Lock()

    def __len__(self) -> int:
        return self._total_frames

    def __getitem__(self, idx: int) -> RGBDFrame:
        frame = self._source[idx]
        if self._pin_memory:
            frame.rgb = frame.rgb.pin_memory()
            frame.depth = frame.depth.pin_memory()
        return frame

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._stop_event.clear()
        self._next_idx = 0

        for i in range(self._num_workers):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()
            self._workers.append(t)

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._workers:
            t.join(timeout=2.0)
        self._workers.clear()
        self._started = False
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def next(self, timeout: Optional[float] = None) -> Optional[RGBDFrame]:
        if not self._started:
            self.start()

        try:
            frame = self._queue.get(timeout=timeout or 5.0)
            return frame
        except queue.Empty:
            with self._idx_lock:
                if self._next_idx >= self._total_frames and self._queue.empty():
                    return None
            return None

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def _worker_loop(self, worker_id: int) -> None:
        while not self._stop_event.is_set():
            with self._idx_lock:
                if self._next_idx >= self._total_frames:
                    return
                idx = self._next_idx
                self._next_idx += 1

            try:
                frame = self._source[idx]

                if self._pin_memory:
                    frame.rgb = frame.rgb.pin_memory()
                    frame.depth = frame.depth.pin_memory()

                self._queue.put(frame)
            except Exception as e:
                print(f"[AsyncRGBDSource] Worker {worker_id} error on frame {idx}: {e}", file=sys.stderr)
                continue

    def close(self) -> None:
        self.stop()
        self._source.close()

    def __del__(self):
        self.stop()
