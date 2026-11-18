"""Gemeinsame Fixtures: der Kleinstfall, an dem jede Einheit von Hand geprüft wird (Ring mit 4 Plätzen, λ = 1, 7 Zustände)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import kpl_chain as K  # noqa: E402


@pytest.fixture(scope="session")
def ring4():
    """Ring mit 4 Plätzen: 7 Zustände (leer, vier Einzelplätze, die Paare {0, 2} und {1, 3}). Handrechnung steht in tests/test_handbeispiel.py."""
    return K.get_ring(4)
