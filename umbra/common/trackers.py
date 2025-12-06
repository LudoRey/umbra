import tracemalloc
import psutil
import threading
import sys
import functools
import time
import numpy as np
import bottleneck as bn
from umbra.common.terminal import cprint
from umbra.common.terminal import ColorTerminalStream

class PsutilMemoryTracker:
    def __init__(self, interval=0.001):
        self.process = psutil.Process()
        self.interval = interval
        self.mem_peak = None
        self.mem_before = None
        self.mem_after = None
        self.running = False

    def __enter__(self):
        self.mem_before = self.process.memory_info().rss
        self.mem_peak = self.mem_before
        self.running = True
        self.thread = threading.Thread(target=self._poll)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.thread.join()

    def _poll(self):
        while self.running:
            mem = self.process.memory_info().rss
            if mem > self.mem_peak:
                self.mem_peak = mem
            self.mem_after = self.process.memory_info().rss
            time.sleep(self.interval)
            
    def get_traced_memory(self):
        return self.mem_after - self.mem_before, self.mem_peak - self.mem_before
    
def track_tracemalloc_memory(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = f(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 1000000:.2f} MB")
        print(f"Peak memory usage   : {peak / 1000000:.2f} MB")
        tracemalloc.stop()
        return result
    return wrapper

def track_psutil_memory(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        with PsutilMemoryTracker() as tracker:
            result = f(*args, **kwargs)
        current, peak = tracker.get_traced_memory()
        print(f"Current memory usage: {current / 1000000:.2f} MB")
        print(f"Peak memory usage   : {peak / 1000000:.2f} MB")
        return result
    return wrapper

def track_time(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f"Elapsed time: {end - start:.2f} seconds")
        return result
    return wrapper

def track_function_name(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        print("-----------")
        cprint(f"{f.__name__}", style="bold")
        return f(*args, **kwargs)
    return wrapper

def track_info(f):
    return track_function_name(track_time(track_psutil_memory(f)))

