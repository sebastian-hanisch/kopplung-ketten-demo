"""Erzeugt `kpl_scaling.json`: gekoppelte Läufe (Heat-bath, Kopplung S, Start im Gipfelpaar) für Ringe von n = 8 bis 40 Plätzen, ohne Übergangsmatrix, dazu die exakte Mischzeit, wo sie sich rechnen lässt.

Aufruf (aus dem Demo-Ordner, mit dem Python des Test-venv):
    python tools/kpl_make_scaling.py             rechnet alle Zeilen neu (Läufe aus SplitMix64 mit festen Zufalls-Seeds; rund 2 bis 4 Minuten, ein Prozess)
    python tools/kpl_make_scaling.py --check     n ≤ 12: die ersten 200 Läufe jeder Zeile neu, E[τ] gegen die Datei im Band (|z| < 4), die exakte Mischzeit neu

Zähleinheiten: Schritte (eine Platzaktualisierung) und Läufe. Zufall nur aus `kpl_rng.SplitMix64`; die Zahlen sind auf jeder Plattform dieselben.
Mischzeit t_mix(1/4) (TV-Abstand d): n ≤ 12 exakt über alle Startzustände, n ≥ 16 nur aus dem Start „alle geraden Plätze belegt“ (dünn besetzte Iteration; das ist eine Untergrenze der Mischzeit,
für n ≤ 12 ist dieser Start der schlechteste, das prüft dieses Skript mit). t_rel = n/g(λ) ist die Formel der Nachbar-Demo zur exakten Mischzeit (gerades n), hier nur als Vergleichsspalte.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import kpl_chain as K  # noqa: E402
import kpl_exact as E  # noqa: E402
import kpl_sim as S  # noqa: E402

OUT = ROOT / "kpl_scaling.json"
ROWS = (
    [(1.0, n, 2000 if n <= 24 else 1000) for n in (8, 12, 16, 20, 24, 32, 40)]
    + [(5.0, n, 2000 if n <= 24 else 1000) for n in (8, 12, 16, 20, 24, 32, 40)]
    + [(20.0, n, 2000 if n <= 12 else 600) for n in (8, 12, 16, 20, 24)]
)
FULL_MAX_N = 12          # bis hierhin wird die Mischzeit aus allen Startzuständen gerechnet
CAP = 50_000_000         # Obergrenze je Lauf (wird nie erreicht)


def g_closed(lam: float) -> float:
    """g(λ) der Formel t_rel = n/g(λ) für gerades n (Heat-bath); q = 1/(1+λ)."""
    q = 1.0 / (1.0 + lam)
    return 4 * q * q / ((1 + 2 * q) + math.sqrt(1 + 4 * q - 4 * q * q))


def seed_for(lam: float, n: int) -> int:
    return 6_000_000 + int(lam * 10) * 1000 + n


def t_mix_top(n: int, lam: float, t_max: int = 100_000):
    """t_mix aus dem Start „alle geraden Plätze belegt“ (Untergrenze), dünn besetzte Vektoriteration."""
    ring = K.get_ring(n)
    pi = ring.pi(lam)
    PT = ring.matrix("heatbath", lam).T.tocsr()
    v = np.zeros(ring.N)
    v[ring.index(ring.top)] = 1.0
    for t in range(1, t_max + 1):
        v = PT @ v
        if 0.5 * np.abs(v - pi).sum() <= 0.25:
            return t
    return None


def t_mix_full(n: int, lam: float):
    ring = K.get_ring(n)
    spec = E.Spectrum(ring.matrix("heatbath", lam), ring.pi(lam))
    return E.first_t(lambda t: spec.d_dbar(t)[0], 0.25)


def run_row(lam: float, n: int, runs: int) -> dict:
    top, bot = K.peak_masks(n)
    taus, cens, steps = S.sample_taus(n, lam, "S", runs, seed_for(lam, n), top, bot, CAP)
    a = np.array(taus, dtype=float)
    ring_n = K.lucas(n)
    row = {
        "lam": lam, "n": n, "N": ring_n, "runs": runs, "seed": seed_for(lam, n), "censored": cens, "steps": steps,
        "E_tau": float(a.mean()), "se": float(a.std(ddof=1) / math.sqrt(len(a))),
        "q75": int(E.sample_quantile(taus, 0.75)), "q90": int(E.sample_quantile(taus, 0.90)),
        "q75_ci": list(E.sample_quantile_ci(taus, 0.75)),
        "t_rel": n / g_closed(lam),
    }
    return row


def exact_part(lam: float, n: int) -> dict:
    out = {"t_mix": None, "t_mix_basis": None}
    if n <= FULL_MAX_N:
        out["t_mix"], out["t_mix_basis"] = t_mix_full(n, lam), "alle Startzustände"
    elif n <= 24:
        out["t_mix"], out["t_mix_basis"] = t_mix_top(n, lam), "Start alle geraden Plätze (Untergrenze)"
    return out


def build(verbose=True) -> dict:
    t0 = time.time()
    rows = []
    for lam, n, runs in ROWS:
        row = run_row(lam, n, runs)
        row.update(exact_part(lam, n))
        rows.append(row)
        if verbose:
            print(f"λ = {lam:g}, n = {n}: E[τ] = {row['E_tau']:.1f}, q75 = {row['q75']}, t_mix = {row['t_mix']} ({time.time() - t0:.0f} s)", flush=True)
    # Prüfung: für n ≤ 12 ist der Start der geraden Plätze der schlechteste Start
    top_equal = all(t_mix_top(n, lam) == t_mix_full(n, lam) for lam in (1.0, 5.0, 20.0) for n in (8, 12))
    return {"meta": {"beschreibung": "Vorgerechnet von tools/kpl_make_scaling.py: gekoppelte Läufe (Heat-bath, Kopplung S, Gipfelpaar) aus SplitMix64, Zähleinheiten Schritte und Läufe",
                     "start_top_gleich_alle_starts_n_bis_12": bool(top_equal)}, "zeilen": rows}


def check(verbose=True) -> list[str]:
    stored = {(r["lam"], r["n"]): r for r in json.loads(OUT.read_text(encoding="utf-8"))["zeilen"]}
    bad = []
    for lam, n, runs in ROWS:
        if n > FULL_MAX_N:
            continue
        row = stored[(lam, n)]
        top, bot = K.peak_masks(n)
        taus, cens, _ = S.sample_taus(n, lam, "S", 200, row["seed"], top, bot, CAP)
        a = np.array(taus, dtype=float)
        se = math.sqrt(a.var(ddof=1) / len(a) + row["se"] ** 2)
        z = (a.mean() - row["E_tau"]) / se
        if abs(z) >= 4 or cens:
            bad.append(f"λ = {lam:g}, n = {n}: z = {z:.2f}")
        if t_mix_full(n, lam) != row["t_mix"]:
            bad.append(f"λ = {lam:g}, n = {n}: t_mix {t_mix_full(n, lam)} gegen {row['t_mix']}")
        if verbose:
            print(f"λ = {lam:g}, n = {n}: z = {z:.2f}", flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        bad = check()
        print("Abweichungen:", len(bad))
        for b in bad:
            print("  ", b)
        sys.exit(1 if bad else 0)
    data = build()
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{OUT.name}: {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
