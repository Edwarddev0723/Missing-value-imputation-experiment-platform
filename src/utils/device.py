"""Device utilities for CPU / MPS / CUDA selection."""

from __future__ import annotations


def get_device(requested: str = "cpu") -> str:
    """Return a valid device string, falling back to CPU if requested device is unavailable."""
    requested = requested.lower()

    if requested == "mps":
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    if requested == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    return "cpu"
