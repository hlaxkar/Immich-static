"""
immich-static — API-First, Zero-Download Static Video Detection and Frame Extraction Toolkit for Immich.
"""

__version__ = "0.1.0"
__author__ = "Immich Static Contributors"

from immich_static.core import (
    Checkpoint,
    detect_video,
    extract_one_frame,
    get_video_metadata,
    resolve_local_video_path,
)
from immich_static.client import ImmichClient
from immich_static.server import run_serve

__all__ = [
    "__version__",
    "Checkpoint",
    "ImmichClient",
    "detect_video",
    "extract_one_frame",
    "get_video_metadata",
    "resolve_local_video_path",
    "run_serve",
]
