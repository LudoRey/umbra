"""Ambient per-task context for the processing pipeline.

Pipeline code reads its progress handlers from here: ``checkstate`` (yields to
the runner to pause/abort) and the ``image`` sink (intermediate images for
display). A runner (the GUI ``Task``) binds real handlers for the duration of a
run via :func:`bind`; outside a run every handler is a no-op, so the pipeline
works as a plain library (CLI, tests) with zero setup.

Handlers are stored in thread-local state: each task runs on its own worker
thread, so concurrent tasks never clobber each other.
"""
from contextlib import contextmanager
import threading


def _noop(*_args, **_kwargs):
    pass


class _Context(threading.local):
    # threading.local subclass: __init__ runs once per thread on first access,
    # so every worker thread starts with fresh no-op handlers.
    def __init__(self):
        self.checkstate = _noop
        self.image = _noop


_ctx = _Context()


# --- pipeline-facing API: call these from anywhere in umbra.* ---
def checkstate() -> None:
    """Yield to the runner so it can pause or abort the task."""
    _ctx.checkstate()


def emit_image(img) -> None:
    """Hand an intermediate image to the runner for display."""
    _ctx.image(img)


# --- runner-facing API: bind handlers for one run ---
@contextmanager
def bind(*, checkstate=None, image=None):
    """Bind handlers for the duration of a ``with`` block, restoring them after.

    Overwrites every slot on entry (so a handler left over from a previous task
    on a reused worker thread cannot bleed through) and restores the previous
    handlers on exit.
    """
    prev = (_ctx.checkstate, _ctx.image)
    _ctx.checkstate = checkstate or _noop
    _ctx.image = image or _noop
    try:
        yield
    finally:
        _ctx.checkstate, _ctx.image = prev
