"""Exakte Größen: Abstände d(t) und d̄(t), Mischzeiten in beiden Zähleinheiten, Satz 5, Quantile und Band der Schätzung, zwei Kontrollfälle (Hyperwürfel, drei Zustände).

Definitionen (Zähleinheit Schritte):
* d(t) = max_x TV(P^t(x,·), π)             — Levin/Peres/Wilmer; **TV-Mischzeit** t_mix(ε) = min{t : d(t) ≤ ε}, Standard ε = 1/4;
* d̄(t) = max_{x,y} TV(P^t(x,·), P^t(y,·))   — Propp/Wilson; **t̄_mix(1/4)** und **T_mix(1/e)** = min{t : d̄(t) ≤ 1/e}.
Es gilt d(t) ≤ d̄(t) ≤ 2·d(t). Die Kopplungsschranke t_c(ε) = min{t : P(τ > t) ≤ ε} ist das (1 − ε)-Quantil der Treffzeit des Gipfelpaars.

Die Dichten kommen aus der Spektralzerlegung der symmetrisierten Übergangsmatrix (nur für kleine Ringe, N ≤ ~2500); Eigene Neuschreibung, kein Import."""

import math
from fractions import Fraction as F

import numpy as np
import scipy.sparse as sp
from scipy.stats import binom

from kpl_chain import popcount


class Spectrum:
    """Spektralzerlegung einer reversiblen Kette: P^t aus Eigenwerten (dicht, N klein)."""

    def __init__(self, P, pi):
        Pd = P.toarray() if sp.issparse(P) else np.asarray(P)
        self.pi = pi
        self.s = np.sqrt(pi)
        S = (self.s[:, None] * Pd) / self.s[None, :]
        ev, U = np.linalg.eigh(0.5 * (S + S.T))
        self.ev, self.U = ev[::-1], U[:, ::-1]

    def power(self, t: int):
        lt = np.sign(self.ev) ** (t % 2) * np.abs(self.ev) ** t
        M = (self.U * lt) @ self.U.T
        return M * (self.s[None, :] / self.s[:, None])

    def d_dbar(self, t: int) -> tuple[float, float]:
        """(d(t), d̄(t)) exakt (bis auf Gleitkomma): maximaler TV-Abstand zu π bzw. zweier Zeilen von P^t."""
        Pt = self.power(t)
        d = 0.5 * np.abs(Pt - self.pi[None, :]).sum(1).max()
        N = Pt.shape[0]
        db = 0.0
        step = max(1, 4_000_000 // (N * N))
        for a in range(0, N, step):
            blk = Pt[a:a + step, None, :] - Pt[None, :, :]
            db = max(db, 0.5 * np.abs(blk).sum(2).max())
        return float(d), float(db)


def first_t(fun, eps: float, t_cap: int = 1 << 40):
    """Kleinstes t ≥ 1 mit fun(t) ≤ eps (fun nichtwachsend): erst verdoppeln, dann halbieren; `None` über t_cap."""
    t = 1
    while fun(t) > eps:
        t *= 2
        if t > t_cap:
            return None
    lo, hi = t // 2, t
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fun(mid) > eps:
            lo = mid
        else:
            hi = mid
    return hi


def mixing_times(spec: Spectrum, eps=0.25, eps_e: float | None = None):
    """(t_mix(ε) aus d, t̄_mix(ε) aus d̄, T_mix(1/e) aus d̄)."""
    e_ = 1 / math.e if eps_e is None else eps_e
    return (first_t(lambda t: spec.d_dbar(t)[0], eps),
            first_t(lambda t: spec.d_dbar(t)[1], eps),
            first_t(lambda t: spec.d_dbar(t)[1], e_))


def longest_chain(ring) -> int:
    """Länge ℓ der längsten Kette der Halbordnung x ≤ y (gerade Plätze: x ⊆ y, ungerade Plätze: y ⊆ x), Levin/Peres/Wilmer Beispiel 22.10; nur für gerades n."""
    if ring.n % 2:
        raise ValueError("Die Ordnung gibt es nur für gerades n")
    EVEN = sum(1 << v for v in range(0, ring.n, 2))
    ODD = sum(1 << v for v in range(1, ring.n, 2))
    e, o = ring.S & EVEN, ring.S & ODD
    A = (((e[:, None] & ~e[None, :]) == 0) & ((o[None, :] & ~o[:, None]) == 0))
    f = popcount(e) - popcount(o)          # (#gerade − #ungerade): wächst längs der Ordnung
    order = np.argsort(f, kind="stable")
    L = np.zeros(ring.N, dtype=np.int64)
    for j in order:
        below = np.flatnonzero(A[:, j])
        below = below[below != j]
        if len(below):
            L[j] = L[below].max() + 1
    return int(L.max())


def grid_points(tmax: int) -> list[int]:
    """Stützstellen für d, d̄ und P(τ > t): alle t = 1 … 40, danach geometrisch (Faktor 1,25), dazu tmax."""
    g = set(range(1, min(tmax, 40) + 1))
    t = 40.0
    while t < tmax:
        t *= 1.25
        g.add(int(t))
    g.add(tmax)
    return sorted(x for x in g if x <= tmax)


# --- Quantile, Schätzband ---------------------------------------------------------------------------------------------------------------------

def quantiles(surv, levels=(0.25, 0.5, 0.75, 0.9, 0.99)):
    """Quantile der Treffzeit aus der Überlebenskurve (surv[t] = P(τ > t)): kleinstes t mit P(τ ≤ t) ≥ Niveau."""
    cdf = 1.0 - np.asarray(surv)
    return {lv: int(np.flatnonzero(cdf >= lv - 1e-12)[0]) for lv in levels}


def sample_rank(R: int, level: float = 0.75) -> int:
    """Rang des Stichprobenquantils: die ⌈level·R⌉-te kleinste Zahl (kleinstes t, bei dem mindestens der Anteil `level` der Läufe fertig ist)."""
    return max(1, math.ceil(level * R - 1e-12))


def sample_quantile(values, level: float = 0.75):
    k = sample_rank(len(values), level)
    return sorted(values)[k - 1]


def quantile_estimate_cdf(surv, R: int, level: float = 0.75):
    """P(q̂_R ≤ t) für t = 0, 1, … : das Stichprobenquantil q̂_R aus R unabhängigen Läufen ist höchstens t, wenn mindestens k = ⌈level·R⌉ Läufe höchstens t brauchen,
    also P = P(Binomial(R, F(t)) ≥ k) mit F(t) = 1 − P(τ > t). Exakt (Ordnungsstatistik), keine Simulation."""
    Ft = 1.0 - np.asarray(surv, dtype=float)
    k = sample_rank(R, level)
    return binom.sf(k - 1, R, np.clip(Ft, 0.0, 1.0))


def quantile_estimate_band(surv, R: int, level: float = 0.75, lo: float = 0.05, hi: float = 0.95):
    """(5-%-Wert, 95-%-Wert) des Stichprobenquantils q̂_R aus R Läufen (kleinste t mit P(q̂ ≤ t) ≥ lo bzw. hi); `None`, wenn die Kurve vorher endet."""
    cdf = quantile_estimate_cdf(surv, R, level)

    def first(q):
        idx = np.flatnonzero(cdf >= q - 1e-12)
        return int(idx[0]) if len(idx) else None
    return first(lo), first(hi)


def prob_estimate_below(surv, R: int, t_mark: int, level: float = 0.75) -> float:
    """P(q̂_R < t_mark) = P(q̂_R ≤ t_mark − 1): Wahrscheinlichkeit, dass die Schätzung unter der Marke liegt."""
    cdf = quantile_estimate_cdf(surv, R, level)
    return float(cdf[t_mark - 1]) if t_mark - 1 < len(cdf) else 1.0


def draw_tau_from_survival(surv, u: float) -> int:
    """Inversionsverfahren: kleinstes t mit F(t) = 1 − surv[t] ≥ u (u in [0, 1)); `len(surv)`, wenn die gespeicherte Kurve vorher endet."""
    cdf = 1.0 - np.asarray(surv, dtype=float)
    i = int(np.searchsorted(cdf, u, side="left"))
    return i


def draw_taus_from_survival(surv, uniforms):
    """Vektorisierte Fassung des Inversionsverfahrens: eine Zahl je Gleichverteilte in `uniforms`."""
    cdf = 1.0 - np.asarray(surv, dtype=float)
    return np.searchsorted(cdf, np.asarray(uniforms, dtype=float), side="left")


def sample_quantile_ci(values, level: float = 0.75, conf: float = 0.95):
    """Verteilungsfreies Band für das Quantil aus einer Stichprobe: (X_(l), X_(u)) mit l, u aus der Binomialverteilung; `None` an der Seite, die für kleine R nicht existiert."""
    R = len(values)
    xs = sorted(values)
    a = (1 - conf) / 2
    l = int(binom.ppf(a, R, level))
    u = int(binom.ppf(1 - a, R, level)) + 1
    return (xs[l - 1] if l >= 1 else None), (xs[u - 1] if u <= R else None)


# --- Kontrollfall Hyperwürfel (Sammler) --------------------------------------------------------------------------------------------------------

def collector_survival(m: int, t_max: int):
    """P(τ > t) der Sammlerkopplung auf dem Hyperwürfel {0,1}^m (Koordinate wählen, auf ein faires Bit setzen; gleiche Koordinate, gleiches Bit)
    vom Paar (alle 0, alle 1): τ ist die Zeit, bis alle m Koordinaten gewählt sind. Exakt über die Zählkette der noch nicht gewählten Koordinaten."""
    v = np.zeros(m + 1)
    v[m] = 1.0
    j = np.arange(m + 1)
    out = [1.0]
    for _ in range(t_max):
        stay = 1 - j / m
        nv = v * stay
        nv[:-1] += v[1:] * (j[1:] / m)
        v = nv
        out.append(float(1.0 - v[0]))
    return np.array(out)


def hypercube_row(m: int):
    """t_mix(1/4) aus d, t̄_mix(1/4) aus d̄ und t_c(1/4) aus der Sammlerkopplung für den Hyperwürfel mit m Koordinaten (exakt über die Zählkette der Einsen) plus E[τ] = m·H_m.

    Start alle 0 (für d: nach Symmetrie der schlechteste Start; die Prüfung steht in tests/test_oracle_kopplung_ketten.py), für d̄ alle 0 gegen alle 1."""
    j = np.arange(m + 1)
    pi = np.array([math.comb(m, int(i)) for i in j], dtype=float) / 2.0 ** m
    a = np.zeros(m + 1)
    a[0] = 1.0
    b = np.zeros(m + 1)
    b[m] = 1.0
    t_mix = t_bar = None
    t = 0
    while t_mix is None or t_bar is None:
        t += 1
        a = _cube_step(a, m)
        b = _cube_step(b, m)
        if t_mix is None and 0.5 * np.abs(a - pi).sum() <= 0.25:
            t_mix = t
        if t_bar is None and 0.5 * np.abs(a - b).sum() <= 0.25:
            t_bar = t
    surv = collector_survival(m, 8 * m + 60)
    while surv[-1] > 0.25:
        surv = collector_survival(m, 2 * (len(surv) - 1))
    t_c = int(np.flatnonzero(surv <= 0.25)[0])
    e_tau = m * sum(1.0 / i for i in range(1, m + 1))
    return {"m": m, "t_mix": t_mix, "t_bar": t_bar, "t_c": t_c, "E_tau": e_tau}


def _cube_step(v, m):
    """Ein Schritt der Zählkette der Einsen auf {0,1}^m: Koordinate gleichverteilt wählen, auf ein faires Bit setzen.
    Von j Einsen aus: nach oben mit Wkt. (m − j)/(2m), nach unten mit Wkt. j/(2m), sonst bleiben."""
    j = np.arange(m + 1)
    stay = 1.0 - (m - j) / (2 * m) - j / (2 * m)
    out = v * stay
    out[1:] += v[:-1] * ((m - j[:-1]) / (2 * m))
    out[:-1] += v[1:] * (j[1:] / (2 * m))
    return out


# --- Kontrollfall drei Zustände (exakte Brüche) ------------------------------------------------------------------------------------------------

def three_state_example(k_max: int = 10):
    """Kette auf {0, 1, 2}: Münze Kopf: +1, Zahl: −1, am Rand bleiben (Levin/Peres/Wilmer, Bemerkung 25.1). π = (1/3, 1/3, 1/3).
    Kopplung: gleiche Münze, Start oben (2) und unten (0); ℓ = 2 (längste Kette der Ordnung 0 < 1 < 2). Alle Größen in exakten Brüchen für k = 1 … k_max."""
    P3 = [[F(1, 2), F(1, 2), F(0)], [F(1, 2), F(0), F(1, 2)], [F(0), F(1, 2), F(1, 2)]]
    pi3 = [F(1, 3)] * 3
    ell = 2

    def tv(p, q):
        return sum(abs(a - b) for a, b in zip(p, q)) / 2

    Pt = [[F(int(i == j)) for j in range(3)] for i in range(3)]
    d, db = {}, {}
    for t in range(1, k_max + 1):
        Pt = [[sum(Pt[i][k] * P3[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        d[t] = max(tv(Pt[i], pi3) for i in range(3))
        db[t] = max(tv(Pt[i], Pt[j]) for i in range(3) for j in range(i + 1, 3))
    step = lambda x, c: min(x + 1, 2) if c else max(x - 1, 0)
    dist = {(2, 0): F(1)}
    tail = [F(1)]
    for _ in range(k_max):
        nd = {}
        for (x, y), p in dist.items():
            for c in (True, False):
                a, b = step(x, c), step(y, c)
                if a != b:
                    nd[(a, b)] = nd.get((a, b), 0) + p / 2
        dist = nd
        tail.append(sum(dist.values()))
    rows = [{"k": k, "tail": tail[k], "tail_over_ell": tail[k] / ell, "d": d[k], "dbar": db[k]} for k in range(1, k_max + 1)]
    # Mittel: Σ_k P(T > k); für k ≥ 1 ist tail[k] = 2^(1−k), der Rest der geometrischen Reihe ab k_max + 1 ist tail[k_max]
    mean = sum(tail[: k_max + 1]) + tail[k_max]
    return {"ell": ell, "rows": rows, "E_tau": mean}
