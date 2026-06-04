"""Supported file extensions, grouped by image format.

This is the public source of truth for which file types the image I/O layer
handles; consumers filter against these sets (e.g. ``imageio.extensions.FITS``).
"""

FITS = frozenset({".fits", ".fit"})

RAW = frozenset({
    ".3fr", ".ari", ".arw", ".bay", ".braw", ".cr2", ".cr3", ".crw", ".dcr",
    ".dng", ".erf", ".fff", ".iiq", ".k25", ".kdc", ".mef", ".mos", ".mrw",
    ".nef", ".nrw", ".orf", ".pef", ".raf", ".raw", ".rw2", ".rwl", ".sr2",
    ".srf", ".srw", ".x3f",
})

BITMAP = frozenset({
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
})

SUPPORTED = FITS | RAW | BITMAP
