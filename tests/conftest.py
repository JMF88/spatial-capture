"""Shared test helpers.

Several pipeline stages live in script-named files (e.g. `01_extract_frames.py`)
that can't be imported normally, so we load them by path via importlib.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def load_module():
    def _load(relpath: str, name: str):
        spec = importlib.util.spec_from_file_location(name, REPO / relpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    return _load
