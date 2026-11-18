"""Auswertung der vier Experimente und die Verdikte der App (reine Funktionen: (Art, Text), die App ruft st.<Art>(Text)).

Jede Zahl im Text eines Verdikts kommt aus einer exakten Rechnung, aus der vorgerechneten Datei oder aus dem Lauf, der gerade angezeigt wird; `tests/test_verdicts.py` und
`tests/test_claims.py` prüfen sie."""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import kpl_chain as K
import kpl_constants as C
import kpl_exact as E
import kpl_format as F
import kpl_pairs as P
import kpl_reference as REF
import kpl_sim as S
from kpl_rng import SplitMix64

TOL = 1e-9    # Toleranz der Ungleichungsprüfungen; die vorgerechnete Datei speichert 12 gültige Stellen

# ======================================================================================================================================
# Experiment 1: zwei Kopien, ein gemeinsamer Zufall
# ======================================================================================================================================


@dataclass(frozen=True)
class E1Settings:
    n: int = C.DEFAULT_E1_N
    lam: float = C.DEFAULT_E1_LAM
    coupling: str = C.DEFAULT_E1_KOPPLUNG
    start: str = C.DEFAULT_START
    seed: int = C.DEFAULT_SEED
    x_text: str = C.DEFAULT_E1_X
    y_text: str = C.DEFAULT_E1_Y


@dataclass(frozen=True)
class E1Result:
    settings: E1Settings
    status: str                   # ok | fehler | gleich | zu_gross | nie | ohne_treffen
    message: str
    x0: int | None = None
    y0: int | None = None
    run: S.Run | None = None
    exact: dict | None = None     # {"surv", "mean", "quant", "source", "min_tau", "never"} oder None
    is_peak: bool = False


def resolve_start(s: E1Settings) -> tuple[int | None, int | None, str]:
    """Startpaar (Bitmasken) aus den Einstellungen; bei fehlerhafter Eingabe (None, None, Meldung)."""
    if s.n < 4:
        return None, None, "Der Ring braucht mindestens 4 Plätze."
    if s.start == "Gipfelpaar":
        top, bot = K.peak_masks(s.n)
        return top, bot, ""
    if s.start == "zwei zufällige Belegungen":
        x, y = S.random_pair(s.seed + 7 * s.n, s.n)
        return x, y, ""
    x, mx = S.parse_state(s.x_text, s.n)
    if x is None:
        return None, None, "Kette X: " + mx
    y, my = S.parse_state(s.y_text, s.n)
    if y is None:
        return None, None, "Kette Y: " + my
    return x, y, ""


@lru_cache(maxsize=48)
def exact_pair_distribution(n: int, lam: float, coupling: str, x0: int, y0: int):
    """Exakte Verteilung von τ für ein Startpaar (n ≤ EXAKT_LIVE_MAX_N, live aus der Paarkette) oder aus der vorgerechneten Datei (n = 10, Kopplung S, Gipfelpaar).

    Rückgabe: dict mit `never` (kein Treffen möglich), `surv` (P(τ > t), t = 0, 1, …), `mean`, `quant`, `source`; oder None, wenn keine exakte Verteilung vorliegt."""
    if x0 == y0:
        return None
    if n <= C.EXAKT_LIVE_MAX_N:
        ring = K.get_ring(n)
        pc = P.PairChain(ring, coupling, lam)
        st = pc.pair(ring.index(x0), ring.index(y0))
        if pc.never_meets(st):
            return {"never": True, "surv": None, "mean": None, "quant": None, "source": "live"}
        surv, _ = pc.survival(st, t_stop=1e-4, t_cap=80_000)
        return {"never": False, "surv": surv, "mean": pc.mean_times(st), "quant": E.quantiles(surv), "source": "live"}
    if n in C.EINGEFROREN_N and coupling == "S" and (x0, y0) == K.peak_masks(n) and REF.has_cell(n, lam):
        c = REF.cell(n, lam)
        full = REF.full_curve(n, lam)
        quant = {float(k): v for k, v in c["quant"].items()}
        return {"never": False, "surv": full, "cell": c, "mean": c["E_tau"], "quant": quant, "source": "vorgerechnet"}
    return None


def analyse_lauf(s: E1Settings) -> E1Result:
    x0, y0, err = resolve_start(s)
    if err:
        return E1Result(s, "fehler", err)
    if x0 == y0:
        return E1Result(s, "gleich", "Beide Ketten starten im selben Zustand: τ = 0.", x0, y0)
    if s.coupling == "I" and s.n > C.UNABHAENGIG_MAX_N:
        return E1Result(s, "zu_gross", f"Die unabhängige Kopplung I ist nur bis n = {C.UNABHAENGIG_MAX_N} freigegeben: ihre Treffzeit wächst etwa mit der Zahl der Zustände "
                        f"({F.ganz(K.lucas(s.n))} bei n = {s.n}) und überschreitet das Schrittbudget.", x0, y0)
    exact = exact_pair_distribution(s.n, s.lam, s.coupling, x0, y0)
    is_peak = (x0, y0) == K.peak_masks(s.n)
    run = S.run_pair(SplitMix64(s.seed), s.n, s.lam, s.coupling, x0, y0, C.LAUF_MAX_SCHRITTE, record=True)
    if exact is not None and exact["never"]:
        status = "nie"
    elif run.tau is None:
        status = "ohne_treffen"
    else:
        status = "ok"
    return E1Result(s, status, "", x0, y0, run, exact, is_peak)


def position_in_distribution(surv, tau: int) -> float | None:
    """P(τ < tau): Anteil der Läufe, die kürzer brauchen als dieser (aus der exakten Überlebenskurve); `None` ohne Kurve oder jenseits ihres Endes."""
    if surv is None or tau - 1 >= len(surv):
        return None
    return float(1.0 - surv[tau - 1]) if tau >= 1 else 0.0


def verdict_lauf(r: E1Result) -> tuple[str, str]:
    s = r.settings
    if r.status == "fehler":
        return "error", r.message
    if r.status == "gleich":
        return "info", r.message
    if r.status == "zu_gross":
        return "warning", r.message
    name = C.COUPLING_SHORT[s.coupling]
    head = f"n = {s.n}, λ = {F.lam(s.lam)}, Kopplung {name}"
    if r.status == "nie":
        return "warning", (f"{head}: **kein Treffen von diesem Startpaar aus**, nach {F.ganz(r.run.steps)} Schritten sind die Ketten immer noch verschieden. Das ist exakt gerechnet: "
                           "kein erreichbares Paar hat einen Übergang auf die Diagonale. Die Kette selbst mischt trotzdem (Experiment 3); „kein Treffen“ sagt über ihre Mischzeit nichts.")
    if r.status == "ohne_treffen":
        extra = (" Bei λ = 1 trifft sich Metropolis mit gemeinsamem u nie (Beweis in Experiment 3); der Lauf bestätigt das nicht, er hört nur auf."
                 if s.coupling == "M" and s.lam == 1.0 else " Das Schrittbudget reicht hier nicht; ob und wann sie sich treffen, sagt dieser Lauf nicht.")
        return "warning", f"{head}: nach {F.ganz(r.run.steps)} Schritten (Obergrenze) haben sich die Ketten nicht getroffen.{extra}"
    tau = r.run.tau
    ex = r.exact
    text = f"{head}: Die Ketten treffen sich nach **τ = {F.ganz(tau)} Schritten**."
    if ex is not None:
        q = ex["quant"]
        pos = position_in_distribution(ex["surv"], tau)
        where = "Startpaar: Gipfelpaar" if r.is_peak else "gewähltes Startpaar"
        text += (f" Exakte Verteilung von τ ({where}; {'vorgerechnet' if ex['source'] == 'vorgerechnet' else 'live gerechnet'}): Viertelquantil {F.ganz(q[0.25])}, Median {F.ganz(q[0.5])}, "
                 f"75-%-Quantil {F.ganz(q[0.75])}, 99-%-Quantil {F.ganz(q[0.99])} Schritte, Mittel {F.dez(ex['mean'])}.")
        if pos is not None:
            text += f" Dieser Lauf liegt über {F.prozent(pos)} der Läufe (P(τ < {F.ganz(tau)}) = {F.dez(pos, 3)})."
        text += " Ein Lauf ist eine Zahl aus dieser Verteilung, kein Wert der Kette."
    else:
        text += " Für diese Einstellung gibt es keine exakte Verteilung (n > 8 oder anderes Paar): ein Lauf ist eine Zahl aus einer unbekannten Verteilung."
    if r.is_peak and s.n % 2 == 1:
        text += " Bei ungeradem n ist das Gipfelpaar nur das Paar der maximalen Belegungen auf den Plätzen 0 bis n − 2: die Ordnung, auf der die Schranke beruht, fehlt."
    return "info", text


# ======================================================================================================================================
# Experiment 2: die Schranke
# ======================================================================================================================================


@dataclass(frozen=True)
class E2Result:
    n: int
    lam: float
    eps: float
    cell: dict
    t_mix: int
    t_bar: int
    T_mix_e: int
    t_c: int
    E_tau: float
    runs_asked: int
    runs_done: int
    taus: tuple
    steps: int
    censored: int


def analyse_schranke(n: int, lam: float, eps: float, runs: int, seed: int) -> E2Result:
    c = REF.cell(n, lam)
    key = f"{eps:g}"
    R = S.run_budget(runs, c["E_tau"], C.SIM_STEP_BUDGET)
    top, bot = K.peak_masks(n)
    taus, cens, steps = S.sample_taus(n, lam, "S", R, seed, top, bot, int(60 * c["E_tau"]) + 1000)
    return E2Result(n, lam, eps, c, c["t_mix_eps"][key], c["t_bar_eps"][key], c["T_mix_e"], c["t_c_eps"][key], c["E_tau"], runs, R, tuple(taus), steps, cens)


def price_stats() -> dict:
    """Der Preis der Kopplung über alle vorgerechneten Zellen (n = 4, 6, 8, 10; 6 Werte von λ; ε = 1/4)."""
    rows = []
    for c in REF.curves()["zellen"].values():
        tm, tb, tc, e = c["t_mix"], c["t_bar"], c["t_c_eps"]["0.25"], c["E_tau"]
        rows.append({"n": c["n"], "lam": c["lam"], "c_over_mix": tc / tm, "c_over_bar": tc / tb, "bar_over_mix": tb / tm, "e_over_mix": e / tm, "four_e_over_mix": 4 * e / tm})
    out = {"cells": len(rows)}
    for k in ("c_over_mix", "c_over_bar", "bar_over_mix", "e_over_mix", "four_e_over_mix"):
        v = np.array([r[k] for r in rows])
        out[k] = (float(v.min()), float(v.max()), float(v.mean()))
    return out


def satz5_stats() -> dict:
    """Satz 5 und die Ungleichungen an allen Stützstellen der vorgerechneten Zellen: Zahl der Punkte und der Verletzungen (Toleranz 1e-9, die Datei speichert 12 Stellen)."""
    n_pts = viol_c = viol_l = viol_r = viol_dd = wrong = 0
    for c in REF.curves()["zellen"].values():
        ell = c["ell"]
        for s_, d, db in zip(c["surv"], c["d"], c["dbar"]):
            n_pts += 1
            viol_c += d > s_ + TOL
            viol_l += s_ / ell > db + TOL
            viol_r += db > s_ + TOL
            viol_dd += (db > 2 * d + TOL) or (d > db + TOL)
            wrong += s_ / ell > d + TOL
    return {"points": n_pts, "d_le_P": int(viol_c), "left": int(viol_l), "right": int(viol_r), "dbar_2d": int(viol_dd), "wrong_d": int(wrong)}


def monotony_stats() -> dict:
    """Das Maximum von P(τ_xy > t) über alle Startpaare gegen das Gipfelpaar: gerades n mit S (24 Zellen), ungerades n mit S (4 Zellen), Metropolis bei λ = 5 (2 Zellen)."""
    cur = REF.curves()
    even = [c["diff_max"] for c in cur["zellen"].values()]
    odd = [v["diff_max"] for v in cur["ungerade"].values()]
    met = [cur["kopplungen"][k]["M"]["diff_max"] for k in ("6|5", "8|5")]
    return {"even": (len(even), max(even)), "odd": (len(odd), max(odd)), "metropolis": (len(met), max(met)), "total": len(even) + len(odd) + len(met),
            "max": max(even + odd + met)}


def verdict_schranke(r: E2Result) -> tuple[str, str]:
    key = f"{r.eps:g}"
    tm, tb, tc = r.t_mix, r.t_bar, r.t_c
    text = (f"n = {r.n}, λ = {F.lam(r.lam)}: t_mix({F.eps(r.eps)}) = {F.ganz(tm)} (Zähleinheit d), t̄_mix({F.eps(r.eps)}) = {F.ganz(tb)} und T_mix(1/e) = {F.ganz(r.T_mix_e)} (Zähleinheit d̄), "
            f"Kopplungsschranke t_c({F.eps(r.eps)}) = {F.ganz(tc)}: **{F.faktor(tc / tm)}-mal t_mix**, {F.faktor(tc / tb)}-mal t̄_mix. ")
    p = price_stats()
    lo, hi, _ = p["c_over_mix"]
    text += (f"In allen {p['cells']} vorgerechneten Zellen (ε = 1/4) liegt t_c beim {F.faktor(lo)}- bis {F.faktor(hi)}-fachen der Mischzeit; die Mittelwert-Schranke 4·E[τ] = {F.ganz(4 * r.E_tau)} "
             f"liegt hier beim {F.faktor(4 * r.E_tau / r.cell['t_mix'])}-fachen (ε = 1/4).")
    if r.eps < 0.25:
        text += " Bei strengerem ε schrumpft der Faktor: die Kopplung liegt nahezu eine feste Zeit, kein fester Faktor, über der Mischzeit."
    return "success", text


# ======================================================================================================================================
# Experiment 3: welche Kopplung, welche Startpunkte
# ======================================================================================================================================


def kopplung_rows(n: int, lam: float, chosen: tuple, mode: str) -> list[dict]:
    block = REF.kopplungen(n, lam)
    rows = []
    for c in C.COUPLINGS:
        if c not in chosen:
            continue
        b = block[c]
        t_c = b["t_c_alle"] if mode == C.E3_MODI[0] else b["t_c_gipfel"]
        rows.append({"coupling": c, "t_mix": b["t_mix"], "never": b["never"], "t_c": t_c, "factor": None if t_c is None else t_c / b["t_mix"], "E_max": b["E_tau_max"],
                     "grid": b["grid"], "curve": b["curve_alle"]})
    return rows


def groesse_rows() -> list[dict]:
    out = []
    for n, g in sorted(REF.curves()["groesse"].items(), key=lambda kv: int(kv[0])):
        out.append({"n": int(n), "N": g["N"], "t_mix": g["t_mix"], "S": g["t_c"]["S"], "C": g["t_c"]["C"], "I": g["t_c"]["I"], "I_over_N": g["t_c"]["I"] / g["N"]})
    return out


def verdict_kopplungen(n: int, lam: float, rows: list[dict]) -> tuple[str, str]:
    if not rows:
        return "info", "Mindestens eine Kopplung wählen."
    parts = []
    for r in rows:
        if r["never"]:
            parts.append(f"{C.COUPLING_SHORT[r['coupling']]}: **trifft sich nie** (obwohl t_mix der Metropolis-Kette {F.ganz(r['t_mix'])} ist)")
        else:
            parts.append(f"{C.COUPLING_SHORT[r['coupling']]}: {F.ganz(r['t_c'])} ({F.faktor(r['factor'])}-fach)")
    base = next(r for r in rows if r["coupling"] != "M") if any(r["coupling"] != "M" for r in rows) else None
    head = f"n = {n}, λ = {F.lam(lam)}: " + (f"t_mix(1/4) = {F.ganz(base['t_mix'])} der Heat-bath-Kette; " if base else "") + "Schranke t_c(1/4) je Kopplung: " + "; ".join(parts) + ". "
    tail = ("Die Treffzeit gehört zur Kopplung, nicht zur Kette: die Schranke hängt an der gewählten Kopplung, und eine Kopplung kann leer bleiben, ohne dass die Kette "
            "schlecht mischt.")
    return "info", head + tail


def verdict_startpaare() -> tuple[str, str]:
    m = monotony_stats()
    return "info", (f"Das Gipfelpaar war in allen {m['total']} gemessenen Zellen das schlechteste Startpaar ({m['even'][0]} Zellen Kopplung S bei geradem n, {m['odd'][0]} Zellen S bei ungeradem n, "
                    f"{m['metropolis'][0]} Zellen Metropolis bei λ = 5; größter Unterschied zum Maximum über alle Paare {m['max']:.1e}). Bewiesen ist das nur mit Monotonie (gerades n, Kopplung S); "
                    "gemessen gilt es in diesen Zellen auch ohne.")


# ======================================================================================================================================
# Experiment 4: ohne Matrix
# ======================================================================================================================================


@dataclass(frozen=True)
class E4Result:
    n: int
    lam: float
    row: dict | None
    runs_asked: int
    runs_done: int
    taus: tuple
    steps: int
    mean: float | None
    se: float | None
    q75: int | None
    q75_ci: tuple | None


def tv_lower_bound_sample(N: int, samples: int) -> float:
    """Bei Gleichverteilung auf N Zuständen hat eine Stichprobe von `samples` Ziehungen höchstens `samples` verschiedene Werte: TV zur Gleichverteilung ≥ 1 − samples/N."""
    return max(0.0, 1.0 - samples / N)


def analyse_matrixfrei(n: int, lam: float, runs: int, seed: int) -> E4Result:
    row = REF.scaling_row(lam, n)
    if row is None:
        return E4Result(n, lam, None, runs, 0, (), 0, None, None, None, None)
    R = S.run_budget(runs, row["E_tau"], C.SIM_STEP_BUDGET)
    top, bot = K.peak_masks(n)
    taus, cens, steps = S.sample_taus(n, lam, "S", R, seed, top, bot, int(60 * row["E_tau"]) + 1000)
    a = np.array(taus, dtype=float)
    se = float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else None
    ci = E.sample_quantile_ci(taus, 0.75) if len(taus) >= 2 else None
    return E4Result(n, lam, row, runs, R, tuple(taus), steps, float(a.mean()), se, int(E.sample_quantile(taus, 0.75)), ci)


def scaling_table() -> list[dict]:
    out = []
    for r in REF.scaling_rows():
        n = r["n"]
        out.append({**r, "q75_over_mix": (r["q75"] / r["t_mix"]) if r["t_mix"] else None, "E_over_nln": r["E_tau"] / (n * math.log(n)), "E_over_rel": r["E_tau"] / r["t_rel"]})
    return out


def scaling_stats() -> dict:
    rows = scaling_table()
    known = [r["q75_over_mix"] for r in rows if r["q75_over_mix"] is not None]
    return {"known": len(known), "lo": min(known), "hi": max(known)}


def verdict_matrixfrei(r: E4Result) -> tuple[str, str]:
    if r.row is None:
        return "info", (f"Für n = {r.n} und λ = {F.lam(r.lam)} liegt keine Messung vor (die Läufe wären zu lang: das Schrittbudget reicht nicht, gemessen sind nur die Zeilen der Tabelle).")
    row = r.row
    N = K.lucas(r.n)
    text = (f"n = {r.n}, λ = {F.lam(r.lam)} ({F.ganz(N)} Zustände): in dieser Rechnung {F.ganz(r.runs_done)} Läufe" + (f" (von {F.ganz(r.runs_asked)} gewünschten; das Schrittbudget begrenzt)" if r.runs_done < r.runs_asked else "") +
            f", E[τ] = {F.dez(r.mean, 1)}" + (f" ± {F.dez(r.se, 1)}" if r.se is not None else "") + f", 75-%-Quantil {F.ganz(r.q75)}")
    if r.q75_ci and r.q75_ci[0] is not None and r.q75_ci[1] is not None:
        text += f" (95-%-Band {F.ganz(r.q75_ci[0])} bis {F.ganz(r.q75_ci[1])})"
    text += f". Vorgerechnet aus {F.ganz(row['runs'])} Läufen: q75 = {F.ganz(row['q75'])}, E[τ] = {F.dez(row['E_tau'], 1)} ± {F.dez(row['se'], 1)}. "
    if row["t_mix"] is not None:
        text += (f"Die exakte Mischzeit ist {F.ganz(row['t_mix'])}" + (" (Untergrenze, aus dem Start der geraden Plätze)" if row["t_mix_basis"] != "alle Startzustände" else "") +
                 f": q75 liegt beim {F.faktor(row['q75'] / row['t_mix'])}-fachen. ")
    else:
        text += "Die Mischzeit ist hier nicht bekannt, die Schranke gilt trotzdem (für diese Kopplung, in Schritten). "
    if r.lam == 1.0:
        text += (f"Eine Stichprobe von 10⁵ Ziehungen aus dem Gleichgewicht hätte zur Gleichverteilung einen TV-Abstand von mindestens {F.dez(tv_lower_bound_sample(N, 10**5), 4)}.")
    return "info", text


# ======================================================================================================================================
# Experiment 4, zweites Panel: wie viele Läufe?
# ======================================================================================================================================


@dataclass(frozen=True)
class LaeufeResult:
    R: int
    reps: int
    t_mix: int
    t_c: int
    band: tuple
    p_below_mix: float
    p_below_c: float
    estimates: tuple              # q̂ je Wiederholung (aus der exakten Verteilung gezogen)
    n_below_mix: int              # Zahl der Wiederholungen mit q̂ unter der Mischzeit


def analyse_laeufe(R: int, reps: int, seed: int) -> LaeufeResult:
    c = REF.cell(C.E4B_N, C.E4B_LAM)
    surv = REF.full_curve(C.E4B_N, C.E4B_LAM)
    t_mix = c["t_mix"]
    t_c = P.tail_quantile(surv, 0.25)
    band = E.quantile_estimate_band(surv, R)
    rng = SplitMix64(seed)
    us = np.array([rng.random() for _ in range(R * reps)]).reshape(reps, R)
    taus = E.draw_taus_from_survival(surv, us.ravel()).reshape(reps, R)
    k = E.sample_rank(R)
    est = [int(v) for v in np.partition(taus, k - 1, axis=1)[:, k - 1]]
    return LaeufeResult(R, reps, t_mix, t_c, band, E.prob_estimate_below(surv, R, t_mix), E.prob_estimate_below(surv, R, t_c), tuple(est),
                        sum(1 for e in est if e < t_mix))


def wk_text(p: float) -> str:
    """Wahrscheinlichkeit als Prozentangabe; unter 0,01 % steht „unter 0,01 %“ statt einer Null."""
    return "unter 0,01 %" if p < 1e-4 else F.prozent(p, 2)


def verdict_laeufe(r: LaeufeResult) -> tuple[str, str]:
    lo, hi = r.band
    return "info", (f"n = 10, λ = 20: exakte Mischzeit {F.ganz(r.t_mix)}, Kopplungsschranke t_c = {F.ganz(r.t_c)}. Das aus R = {r.R} Läufen geschätzte 75-%-Quantil von τ streut exakt von {F.ganz(lo)} bis {F.ganz(hi)} "
                    f"(5 % bis 95 %) und liegt mit Wahrscheinlichkeit {wk_text(r.p_below_mix)} unter der Mischzeit (in {F.ganz(r.reps)} gezogenen Wiederholungen: {r.n_below_mix} von {F.ganz(r.reps)}). "
                    f"Unter der Mischzeit liegt die Schätzung hier nur selten; Deutung (nicht gemessen): die exakte Schranke ist mit {F.ganz(r.t_c)} mehr als das Doppelte der Mischzeit {F.ganz(r.t_mix)}. "
                    "Bei zwei scharfen Gipfeln (λ = 100) ist das nicht gemessen.")


# ======================================================================================================================================
# Kontrollfälle
# ======================================================================================================================================


@lru_cache(maxsize=1)
def hypercube_table() -> list[dict]:
    rows = []
    for m in (8, 12, 50, 100, 400):
        r = E.hypercube_row(m)
        rows.append({**r, "c_over_mix": r["t_c"] / r["t_mix"], "c_over_bar": r["t_c"] / r["t_bar"]})
    return rows


@lru_cache(maxsize=1)
def three_state() -> dict:
    return E.three_state_example(10)


def verdict_satz5() -> tuple[str, str]:
    ts = three_state()
    r1 = ts["rows"][0]
    st = satz5_stats()
    return "info", (f"Drei Zustände (ℓ = 2): P(T > 1)/ℓ = {F.bruch(r1['tail_over_ell'])} liegt **über d(1) = {F.bruch(r1['d'])}** und ist gleich d̄(1) = {F.bruch(r1['dbar'])}: gegen d statt d̄ geprüft, "
                    f"sähe Satz 5 verletzt aus. Am Ring (ℓ = n) zeigt sich der Unterschied in {st['wrong_d']} von {F.ganz(st['points'])} Stützstellen (n = 4 bis 10, λ = 0,5 bis 20); "
                    f"der Satz selbst wird dort in {st['left'] + st['right']} Stützstellen verletzt.")
