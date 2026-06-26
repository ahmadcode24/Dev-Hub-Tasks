"""
utils.py
--------
Utility functions for logging, path management, and report generation.
"""

import os
from datetime import datetime


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def generate_timestamp() -> str:
    """Return a formatted timestamp string for logging."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def print_section_header(title: str, width: int = 60) -> None:
    """Print a styled section header for notebook readability."""
    print("═" * width)
    print(title.upper().center(width))
    print("═" * width)
