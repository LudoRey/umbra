"""Supported file extensions, grouped by image format.

This is the public source of truth for which file types the image I/O layer
handles; consumers filter against these sets (e.g. ``imageio.extensions.FITS``).
"""

FITS = frozenset({".fits", ".fit"})

# Restricted to formats whose metadata we know we can read: CR3, whose EXIF the exif
# backend locates explicitly, and raws headed by a TIFF, whose EXIF exifread locates
# unaided. Formats we could not confirm are left out rather than guessed at; add one
# back once a sample shows its EXIF comes through.
RAW = frozenset({
    ".arw", ".cr2", ".cr3", ".dng", ".nef", ".nrw", ".orf", ".pef", ".rw2",
    ".rwl",
})

BITMAP = frozenset({
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
})

SUPPORTED = FITS | RAW | BITMAP
