"""Erzeugt `kpl_curves.json`: die vorgerechneten exakten Größen für Experiment 2 und 3 (Ring mit n = 4, 6, 8, 10 und 6 Werten von λ).

Aufruf (aus dem Demo-Ordner, mit dem Python des Test-venv):
    python tools/kpl_make_curves.py            schreibt kpl_curves.json neu (alle Zellen, rund 1 bis 2 Minuten, n = 10 überwiegt)
    python tools/kpl_make_curves.py --check    rechnet n ≤ 8 neu und vergleicht mit der Datei (Exit 1 bei Abweichung)
    python tools/kpl_make_curves.py --check --lang   rechnet auch n = 10 neu (rund 1 bis 2 Minuten)

Alles exakt (bis auf Gleitkomma): Übergangsmatrix, d(t), d̄(t) aus der Spektralzerlegung, P(τ > t) und E[τ] aus der Paarkette der Kopplung. Heat-bath, Kopplung S (gemeinsamer Platz und
gemeinsame Münze), Start im Gipfelpaar, wenn nicht anders gesagt. Zähleinheit: Schritte. Die Datei ist vorgerechnet und wird in App und README so gekennzeichnet.
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
import kpl_pairs as P  # noqa: E402

OUT = ROOT / "kpl_curves.json"
LAMS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
NS = (4, 6, 8, 10)
EPSS = (0.25, 0.1, 0.01, 0.001)
KOPPL_CELLS = ((6, 1.0), (6, 5.0), (8, 1.0), (8, 5.0))
ODD_CELLS = ((7, 1.0), (7, 5.0), (9, 1.0), (9, 5.0))
FULL_CELL = (10, 20.0)


def r12(x):
    return None if x is None else float(f"{x:.12g}")


def key(n, lam):
    return f"{n}|{lam:g}"


def compute_cell(n: int, lam: float, full: bool = False) -> dict:
    """Alle exakten Größen einer Zelle (n, λ): Heat-bath, Kopplung S, Start im Gipfelpaar (und das Maximum über alle Paare)."""
    ring = K.get_ring(n)
    pi = ring.pi(lam)
    spec = E.Spectrum(ring.matrix("heatbath", lam), pi)
    tmix, tbar, tmix_e = E.mixing_times(spec)
    top, bot = ring.index(ring.top), ring.index(ring.bot)
    pc = P.PairChain(ring, "S", lam)
    st = pc.pair(top, bot)
    T = int(10 * tbar) + 20
    mx, ex = pc.curve(T, ext=st)
    t_last = len(mx) - 1
    grid = E.grid_points(t_last)
    dd = [spec.d_dbar(t) for t in grid]
    tmix_eps = {f"{e:g}": E.first_t(lambda t, e=e: spec.d_dbar(t)[0], e) for e in EPSS}
    tbar_eps = {f"{e:g}": E.first_t(lambda t, e=e: spec.d_dbar(t)[1], e) for e in EPSS}
    means = pc.mean_times()
    cell = {
        "n": n, "lam": lam, "N": ring.N,
        "ell": E.longest_chain(ring) if n % 2 == 0 else None,
        "t_mix": tmix, "t_bar": tbar, "T_mix_e": tmix_e,
        "t_mix_eps": tmix_eps, "t_bar_eps": tbar_eps,
        "t_c_eps": {f"{e:g}": P.tail_quantile(ex, e) for e in EPSS},
        "t_c_alle": P.tail_quantile(mx, 0.25),
        "E_tau": r12(float(means[st])), "E_tau_max": r12(float(means.max())),
        "quant": {f"{q:g}": v for q, v in E.quantiles(ex).items()},
        "diff_max": float(np.max(mx - ex)),
        "grid": grid,
        "surv": [r12(float(ex[t])) for t in grid],
        "d": [r12(a) for a, _ in dd],
        "dbar": [r12(b) for _, b in dd],
    }
    if full:
        # vollständige Überlebenskurve (jedes t), bis P(τ > t) unter 1e-4 fällt: Grundlage der exakten Bänder in Experiment 4
        cut = int(np.flatnonzero(ex < 1e-4)[0]) + 1 if (ex < 1e-4).any() else len(ex)
        cell["surv_full"] = [float(f"{v:.8g}") for v in ex[:cut]]
    return cell


def compute_kopplungen(n: int, lam: float) -> dict:
    """Experiment 3: vier Kopplungen derselben Kette, Maximum über ALLE Startpaare und das Gipfelpaar."""
    ring = K.get_ring(n)
    pi = ring.pi(lam)
    spec_h = E.Spectrum(ring.matrix("heatbath", lam), pi)
    spec_m = E.Spectrum(ring.matrix("metropolis", lam), pi)
    tm_h = E.first_t(lambda t: spec_h.d_dbar(t)[0], 0.25)
    tm_m = E.first_t(lambda t: spec_m.d_dbar(t)[0], 0.25)
    top, bot = ring.index(ring.top), ring.index(ring.bot)
    out = {}
    for c in ("S", "C", "I", "M"):
        pc = P.PairChain(ring, c, lam)
        st = pc.pair(top, bot)
        tm = tm_m if c == "M" else tm_h
        T = int(12 * tm) + 50
        mx, ex = pc.curve(T, ext=st)
        never = pc.never_meets(st)
        mean_all = None if never else pc.mean_times()
        grid = E.grid_points(len(mx) - 1)
        out[c] = {
            "t_mix": tm, "never": bool(never),
            "t_c_alle": P.tail_quantile(mx, 0.25), "t_c_gipfel": P.tail_quantile(ex, 0.25),
            "E_tau_max": None if mean_all is None else r12(float(np.max(mean_all))),
            "diff_max": float(np.max(mx - ex)),
            "grid": grid, "curve_alle": [r12(float(mx[t])) for t in grid],
        }
    return out


def compute_groesse(n: int) -> dict:
    """Größe bei λ = 1: Heat-bath, Kopplungen S, C, I über alle Paare (Zustandszahl N = Lucas-Zahl)."""
    ring = K.get_ring(n)
    lam = 1.0
    spec = E.Spectrum(ring.matrix("heatbath", lam), ring.pi(lam))
    tm = E.first_t(lambda t: spec.d_dbar(t)[0], 0.25)
    vals = {}
    for c in ("S", "C", "I"):
        mx, _ = P.PairChain(ring, c, lam).curve(int(40 * tm) + 100)
        vals[c] = P.tail_quantile(mx, 0.25)
    return {"N": ring.N, "t_mix": tm, "t_c": vals}


def compute_odd(n: int, lam: float) -> dict:
    """Ungerades n (keine Ordnung): Kopplung S, das Paar der maximalen Belegungen gegen alle Paare."""
    ring = K.get_ring(n)
    spec = E.Spectrum(ring.matrix("heatbath", lam), ring.pi(lam))
    tm = E.first_t(lambda t: spec.d_dbar(t)[0], 0.25)
    pc = P.PairChain(ring, "S", lam)
    st = pc.pair(ring.index(ring.top), ring.index(ring.bot))
    mx, ex = pc.curve(int(12 * tm) + 50, ext=st)
    return {"n": n, "lam": lam, "t_mix": tm, "t_c_alle": P.tail_quantile(mx, 0.25), "t_c_gipfel": P.tail_quantile(ex, 0.25), "diff_max": float(np.max(mx - ex))}


def build(ns=NS, verbose=True) -> dict:
    t0 = time.time()
    data = {"meta": {
        "beschreibung": "Vorgerechnet von tools/kpl_make_curves.py: exakte Größen des Hard-Core-Rings (Heat-bath, Kopplung S, Start im Gipfelpaar), Zähleinheit Schritte",
        "lam_gitter": list(LAMS), "n": list(NS), "eps": list(EPSS),
    }, "zellen": {}, "kopplungen": {}, "groesse": {}, "ungerade": {}}
    for n in ns:
        for lam in LAMS:
            data["zellen"][key(n, lam)] = compute_cell(n, lam, full=(n, lam) == FULL_CELL)
            if verbose:
                print(f"Zelle n = {n}, λ = {lam:g} fertig ({time.time() - t0:.0f} s)", flush=True)
    for (n, lam) in KOPPL_CELLS:
        data["kopplungen"][key(n, lam)] = compute_kopplungen(n, lam)
    for n in ns:
        data["groesse"][str(n)] = compute_groesse(n)
    for (n, lam) in ODD_CELLS:
        data["ungerade"][key(n, lam)] = compute_odd(n, lam)
    if verbose:
        print(f"fertig nach {time.time() - t0:.0f} s", flush=True)
    return data


def _same(a, b, path, bad, rtol=1e-9):
    if isinstance(a, dict):
        if set(a) != set(b):
            bad.append(f"{path}: Schlüssel verschieden")
            return
        for k in a:
            _same(a[k], b[k], f"{path}/{k}", bad, rtol)
    elif isinstance(a, list):
        if len(a) != len(b):
            bad.append(f"{path}: Länge {len(a)} gegen {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]", bad, rtol)
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or not math.isclose(a, b, rel_tol=rtol, abs_tol=1e-12):
            bad.append(f"{path}: {a} gegen {b}")
    elif a != b:
        bad.append(f"{path}: {a} gegen {b}")


def check(lang: bool = False, verbose=True) -> list[str]:
    """Rechnet n ≤ 8 neu (mit `lang` auch n = 10) und vergleicht mit der Datei; gibt die Liste der Abweichungen zurück."""
    stored = json.loads(OUT.read_text(encoding="utf-8"))
    ns = NS if lang else tuple(n for n in NS if n <= 8)
    fresh = build(ns, verbose=verbose)
    bad = []
    for k, cell in fresh["zellen"].items():
        _same(cell, stored["zellen"][k], f"zellen/{k}", bad)
    for part in ("kopplungen", "groesse", "ungerade"):
        for k, v in fresh[part].items():
            if part == "groesse" and int(k) > 8 and not lang:
                continue
            _same(v, stored[part][k], f"{part}/{k}", bad)
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--lang", action="store_true")
    args = ap.parse_args()
    if args.check:
        bad = check(args.lang)
        print("Abweichungen:", len(bad))
        for b in bad[:20]:
            print("  ", b)
        sys.exit(1 if bad else 0)
    data = build()
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{OUT.name}: {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
