"""Diagnostic plots for the registration pipeline.

Built with the matplotlib object API (never pyplot) so they can be constructed
on a worker thread and handed to a runner via :func:`umbra.common.context.emit_plot`.
Each builder returns a Figure summarising the whole run so far; the caller
re-emits them under a stable label as records accumulate, so a live runner
updates each plot in place.
"""
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from umbra.common import context, fits


@dataclass
class FrameRecord:
    """Per-frame registration results, accumulated over a run for plotting.

    ``rotation`` and ``sun_moon_translation`` are relative to the reference
    frame, so the reference frame itself sits at the origin.
    """
    filename: str
    timestamp: float | None
    moon_center: np.ndarray             # [x, y] in pixels
    moon_radius: float
    inlier_fraction: float              # RANSAC inliers over detected edge pixels
    rotation: float                     # radians
    sun_moon_translation: np.ndarray    # [x, y] in pixels
    is_anchor: bool
    is_reference: bool

    @classmethod
    def from_frame(
        cls, filename, header, moon_center, moon_radius, inlier_fraction,
        rotation, sun_moon_translation, is_anchor, is_reference,
    ) -> "FrameRecord":
        try:
            timestamp = fits.extract_timestamp(header)
        except ValueError:
            timestamp = None
        return cls(
            filename, timestamp, moon_center, moon_radius, inlier_fraction,
            rotation, sun_moon_translation, is_anchor, is_reference,
        )


# Point colours by kind. "other" is any frame that is neither the reference nor
# an anchor; its label depends on the plot. Red is reserved for the bound lines.
_COLOR = {"reference": "C2", "anchor": "C1", "other": "C0"}
_DOT_SIZE = 10


def _kind(record: FrameRecord) -> str:
    if record.is_reference:
        return "reference"
    return "anchor" if record.is_anchor else "other"


def _sorted(records: Sequence[FrameRecord]) -> list[FrameRecord]:
    """Records in x-axis order: by timestamp when all have one, else by name."""
    if all(r.timestamp is not None for r in records):
        return sorted(records, key=lambda r: r.timestamp)
    return sorted(records, key=lambda r: r.filename)


def _x_of(records: Sequence[FrameRecord], all_records: Sequence[FrameRecord]) -> tuple[np.ndarray, str]:
    """X values for ``records``, plus an axis label.

    Elapsed seconds from the reference frame when every frame is timestamped,
    else the frame's index within the full run (so subsets stay aligned).
    """
    if all(r.timestamp is not None for r in all_records):
        ref = next((r.timestamp for r in all_records if r.is_reference), _sorted(all_records)[0].timestamp)
        return np.array([r.timestamp - ref for r in records]), "time since reference (s)"
    order = {id(r): i for i, r in enumerate(_sorted(all_records))}
    return np.array([order[id(r)] for r in records]), "frame index"


def moon_detection_figure(records: Sequence[FrameRecord], radius_bounds: tuple[float, float]) -> Figure:
    """Detected moon radius, centre (x, y) and RANSAC inlier fraction against time,
    one marker per frame.

    ``radius_bounds`` is the plausible-radius acceptance range, drawn as bounds.
    """
    records = _sorted(records)
    x, xlabel = _x_of(records, records)

    fig = Figure(figsize=(7, 9), layout="constrained")
    ax_radius, ax_x, ax_y, ax_inliers = fig.subplots(4, 1, sharex=True)
    lower, upper = radius_bounds
    ax_radius.axhline(lower, color="red", linestyle="--", linewidth=1, zorder=0)
    ax_radius.axhline(upper, color="red", linestyle="--", linewidth=1, zorder=0)
    _scatter_by_kind(ax_radius, records, x, [r.moon_radius for r in records])
    _scatter_by_kind(ax_x, records, x, [r.moon_center[0] for r in records])
    _scatter_by_kind(ax_y, records, x, [r.moon_center[1] for r in records])
    _scatter_by_kind(ax_inliers, records, x, [r.inlier_fraction for r in records])
    ax_radius.set_ylabel("radius (px)")
    ax_x.set_ylabel("center x (px)")
    ax_y.set_ylabel("center y (px)")
    # A proportion is only readable against its full range, so never autoscale it.
    ax_inliers.set_ylim(0, 1)
    ax_inliers.set_ylabel("inlier fraction")
    ax_inliers.set_xlabel(xlabel)
    bound_handle = Line2D([], [], color="red", linestyle="--", linewidth=1, label="plausible range")
    ax_radius.legend(handles=_kind_handles(records, {}) + [bound_handle], loc="best", fontsize="small")
    fig.suptitle("Moon detection")
    return fig


def _scatter_by_kind(ax, records, x, y) -> None:
    """Scatter ``x`` vs ``y`` (both aligned with ``records``), coloured by kind."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    kinds = np.array([_kind(r) for r in records])
    # Draw "other" (the bulk) first so the sparse anchors and reference land on top.
    for kind, color in reversed(_COLOR.items()):
        mask = kinds == kind
        if mask.any():
            ax.scatter(x[mask], y[mask], color=color, s=_DOT_SIZE, zorder=3)


def _kind_handles(records, labels: dict) -> list:
    """Legend handles for the kinds present, each labelled via ``labels`` (which
    overrides the kind name, e.g. ``{"other": "interpolated"}``)."""
    return [
        Line2D([], [], marker="o", linestyle="none", color=_COLOR[k], label=labels.get(k, k))
        for k in _COLOR if any(_kind(r) == k for r in records)
    ]


def anchor_values_figure(records: Sequence[FrameRecord]) -> Figure:
    """Rotation and sun-moon delta against time: anchors measured (with the
    interpolant line through them), other frames interpolated onto that line,
    the reference frame at the origin."""
    records = _sorted(records)
    anchors = [r for r in records if r.is_anchor]
    x, xlabel = _x_of(records, records)
    xa, _ = _x_of(anchors, records)
    line_order = np.argsort(xa)
    anchor_pos = [records.index(a) for a in anchors]

    fig = Figure(figsize=(7, 7), layout="constrained")
    ax_rot, ax_dx, ax_dy = fig.subplots(3, 1, sharex=True)
    series = [
        (ax_rot, np.rad2deg([r.rotation for r in records]), "rotation (deg)"),
        (ax_dx, [r.sun_moon_translation[0] for r in records], "sun−moon Δx (px)"),
        (ax_dy, [r.sun_moon_translation[1] for r in records], "sun−moon Δy (px)"),
    ]
    for ax, y, ylabel in series:
        y = np.asarray(y, dtype=float)
        ax.plot(xa[line_order], y[anchor_pos][line_order], color="0.6", linewidth=1, zorder=1)
        _scatter_by_kind(ax, records, x, y)
        ax.set_ylabel(ylabel)
    ax_dy.set_xlabel(xlabel)
    fig.suptitle("Anchor values")
    ax_rot.legend(handles=_kind_handles(records, {"other": "interpolated"}), loc="best", fontsize="small")
    return fig


def emit(records: Sequence[FrameRecord], radius_bounds: tuple[float, float]) -> None:
    """Emit the registration diagnostic figures to the runner, one per label."""
    context.emit_plot(moon_detection_figure(records, radius_bounds), "Moon detection")
    context.emit_plot(anchor_values_figure(records), "Anchor values")
