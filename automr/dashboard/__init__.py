"""
AutoMR Live Dashboard package.
"""

from .video_runner import run_live_dashboard
from .camera_source import CameraSource

__all__ = [
    "run_live_dashboard",
    "CameraSource",
]