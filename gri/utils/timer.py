# gri/utils/timer.py
import time

class Timer:
    def __init__(self, name=""):
        self.name = name
        self.t0 = 0.0
        self.elapsed = 0.0
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.t0
        if self.name:
            print(f"[{self.name}] {self.elapsed*1000:.2f} ms")
