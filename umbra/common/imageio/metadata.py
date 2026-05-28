from pathlib import Path
from typing import Any

import exifread


def extract_metadata(filepath: Path | str) -> dict[str, Any]:
    """Extract EXIF metadata from an image file."""
    filepath = Path(filepath)
    with filepath.open("rb") as file:
        return exifread.process_file(file, details=False, builtin_types=True)
