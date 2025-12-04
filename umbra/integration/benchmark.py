import tracemalloc
import psutil
import threading
import sys
import functools
import time
import numpy as np
import bottleneck as bn
from umbra.common.utils import cprint
from umbra.common.utils import ColorTerminalStream

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

def announce_function_name(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        print("-----------")
        cprint(f"{f.__name__}", style="bold")
        return f(*args, **kwargs)
    return wrapper

def info(f):
    return announce_function_name(track_time(track_psutil_memory(f)))


@info
def numpy_masked_mean(masked_stack: np.ma.MaskedArray):
    return np.mean(masked_stack, axis=0) # returns a float64 masked array
    
@info
def numpy_masked_std(masked_stack: np.ma.MaskedArray, masked_mean: np.ma.MaskedArray = None):
    return np.std(masked_stack, mean=masked_mean, axis=0) # returns a float64 masked array

@info
def numpy_mean(stack: np.ndarray):
    return np.mean(stack, axis=0)
    
@info
def numpy_std(stack: np.ndarray, mean: np.ndarray = None):
    return np.std(stack, mean=mean, axis=0)

@info
def numpy_nanmean(stack: np.ndarray):
    return np.nanmean(stack, axis=0)

@info
def numpy_nanstd(stack: np.ndarray, mean: np.ndarray = None):
    if mean is None:
        return np.nanstd(stack, axis=0)
    return np.nanstd(stack, mean=mean, axis=0)

@info
def bottleneck_nanmean(stack: np.ndarray):
    return bn.nanmean(stack, axis=0)

@info
def bottleneck_nanstd(stack: np.ndarray):
    return bn.nanstd(stack, axis=0)

def generate_mask(num_images, shape, p):
    mask = np.random.rand(num_images, *shape) < p
    all_true_slices = np.all(mask, axis=0)  # shape: (height, width)
    mask[0, all_true_slices] = False # Unmask the first image for each all true slice
    return mask

def generate_stack(num_images, shape):
    return np.random.rand(num_images, *shape).astype(np.float32)

if __name__ == "__main__":
    sys.stdout = ColorTerminalStream()
    num_images = 10
    shape = (10000, 1000)
    p = 0.5
    
    mask = generate_mask(num_images, shape, p)
    stack = generate_stack(num_images, shape)
    print(f"Mask size: {mask.nbytes / 1000000:.2f} MB")
    print(f"Stack size: {stack.nbytes / 1000000:.2f} MB")
    
    mean = numpy_mean(stack)
    std = numpy_std(stack, mean=mean)
    
    # masked_stack = np.ma.MaskedArray(stack, mask=mask)
    # masked_mean = numpy_masked_mean(masked_stack)
    # masked_std = numpy_masked_std(masked_stack, masked_mean=masked_mean)
    
    stack[mask] = np.nan
    np_mean = numpy_nanmean(stack)
    np_std = numpy_nanstd(stack, mean=np_mean)
    
    bn_mean = bottleneck_nanmean(stack)
    bn_std = bottleneck_nanstd(stack)
    
    assert np.allclose(bn_mean, np_mean)
    assert np.allclose(bn_std, np_std)