"""Vorgerechnete Daten laden (`kpl_curves.json`, `kpl_scaling.json`) und als Funktionen anbieten.

Beide Dateien entstehen aus den Skripten `tools/kpl_make_curves.py` und `tools/kpl_make_scaling.py` (mit `--check`); die App kennzeichnet ihre Zahlen als „vorgerechnet“."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def curves() -> dict:
    return json.loads((ROOT / "kpl_curves.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def scaling() -> dict:
    return json.loads((ROOT / "kpl_scaling.json").read_text(encoding="utf-8"))


def cell(n: int, lam: float) -> dict:
    """Vorgerechnete Zelle (n, λ): Heat-bath, Kopplung S, Gipfelpaar (n = 4, 6, 8, 10; λ aus dem Gitter)."""
    return curves()["zellen"][f"{n}|{lam:g}"]


def has_cell(n: int, lam: float) -> bool:
    return f"{n}|{lam:g}" in curves()["zellen"]


def kopplungen(n: int, lam: float) -> dict:
    return curves()["kopplungen"][f"{n}|{lam:g}"]


def survival_at(c: dict, t: int) -> float:
    """P(τ > t) aus der vorgerechneten Zelle: genau an den Stützstellen, dazwischen linear; bei vorhandener vollständiger Kurve (n = 10, λ = 20) exakt."""
    full = c.get("surv_full")
    if full is not None and t < len(full):
        return full[t]
    if t <= 0:
        return 1.0
    g = c["grid"]
    if t > g[-1]:
        return float(c["surv"][-1])
    return float(np.interp(t, [0] + g, [1.0] + c["surv"]))


def full_curve(n: int, lam: float):
    """Vollständige Überlebenskurve P(τ > t) (jedes t), nur für n = 10, λ = 20 gespeichert."""
    c = cell(n, lam)
    return None if "surv_full" not in c else np.array(c["surv_full"])


def scaling_rows() -> list[dict]:
    return scaling()["zeilen"]


def scaling_row(lam: float, n: int) -> dict | None:
    for r in scaling_rows():
        if r["lam"] == lam and r["n"] == n:
            return r
    return None
