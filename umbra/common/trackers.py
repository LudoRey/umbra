import io
import tracemalloc
import psutil
import threading
import sys
import functools
import time
from umbra.common.terminal import cprint

class PsutilMemoryTracker:
    def __init__(self, interval=0.001):
        self.process = psutil.Process()
        self.interval = interval
        self.mem_peak = None
        self.mem_current = None
        self.running = False

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self._poll)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.thread.join()

    def _poll(self):
        self.mem_peak = self.get_current_memory()
        while self.running:
            self.mem_current = self.get_current_memory()
            if self.mem_current > self.mem_peak:
                self.mem_peak = self.mem_current
            time.sleep(self.interval)

    def get_peak_memory(self):
        return self.mem_peak
    
    def get_current_memory(self):
        return self.process.memory_info().rss
    
class StreamPrefixer:
    def __init__(self, stream: io.TextIOBase, prefix: str):
        self.stream = stream
        self.prefix = prefix
        self.original_write = stream.write
        self.at_line_start = True

    def prefixed_write(self, data: str):
        i = 0
        while i < len(data):
            if self.at_line_start:
                self.original_write(self.prefix)
                self.at_line_start = False
            newline_pos = min(
                [data.find(x, i) if data.find(x, i) != -1 else len(data) for x in ("\n", "\r")]
            )
            if newline_pos == len(data):
                self.original_write(data[i:])
                break
            self.original_write(data[i:newline_pos + 1])
            i = newline_pos + 1
            self.at_line_start = True

    def __enter__(self):
        self.stream.write = self.prefixed_write
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.write = self.original_write
    
def track_tracemalloc_memory(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = f(*args, **kwargs)
        exit, peak = tracemalloc.get_traced_memory()
        cprint(f"Exit memory delta: {exit / 1000000:.2f} MB", color="yellow")
        cprint(f"Peak memory delta: {peak / 1000000:.2f} MB", color="yellow")
        tracemalloc.stop()
        return result
    return wrapper

def track_psutil_memory(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        with PsutilMemoryTracker() as tracker:
            entry = tracker.get_current_memory()
            cprint(f"Entry memory: {entry / 1000000:.2f} MB", color="yellow")
            result = f(*args, **kwargs)
            exit = tracker.get_current_memory()
            peak = tracker.get_peak_memory()
            cprint(f"Exit memory: {exit / 1000000:.2f} MB ({(exit - entry) / 1000000:+.2f} MB)", color="yellow")
            cprint(f"Peak memory: {peak / 1000000:.2f} MB ({(peak - entry) / 1000000:+.2f} MB)", color="yellow")
        return result
    return wrapper

def track_time(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        cprint(f"Elapsed time: {end - start:.2f} seconds", color="yellow")
        return result
    return wrapper

def wrap_output(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        line_width = 40
        cprint("=" * line_width, style="bold")
        with StreamPrefixer(sys.stdout, prefix="| "):
            cprint(f.__name__, style="bold", color="red")
            result = f(*args, **kwargs)
        cprint("=" * line_width, style="bold")
        return result
    return wrapper

def track_info(f):
    return wrap_output(track_time(track_psutil_memory((f))))
