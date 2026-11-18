"""Orakel-Tests: jede Kernroutine gegen einen anderen Rechenweg (ORACLE-PLAYBOOK §2, §5).

* Zustandsraum und Übergangsmatrizen gegen Aufzählen aller 2^n Masken und gegen eine zweite Matrix aus dem Erzeuger (Heat-bath: Platz mal Münze; Metropolis: I + Q/(n·max(1, λ))).
* Die Paarkette (dünn besetzte Matrix Q, Potenzen) gegen eine Vorwärtsrechnung über Wörterbücher aus den Zufallsquellen (Platz, Münze, u) für alle vier Kopplungen.
* Randketten der Paarkerne S, C, M gleich der Übergangsmatrix: jede der vier ist eine Kopplung.
* d(t) und d̄(t) aus der Spektralzerlegung gegen die Matrixpotenz; E[τ] aus dem linearen System gegen die Reihe Σ_t P(τ > t).
* Sammler-Formel (Hyperwürfel) gegen die Zählkette und gegen die Paarkette aus dem Wörterbuch-Rechenweg.
* Simulierte Treffzeiten (SplitMix64) gegen die exakte Überlebenskurve: Kolmogorov-Abstand und Mittel mit |z| < 4.
* Ordnungsstatistik der Schätzung des 75-%-Quantils gegen Aufzählen aller Ausgänge."""

import itertools
import math

import numpy as np
import pytest

import kpl_chain as K
import kpl_exact as E
import kpl_pairs as P
import kpl_sim as S

LAMS = (0.5, 1.0, 2.0, 5.0)


def brute_states(n):
    out = []
    for m in range(1 << n):
        if all(not (m >> v & 1 and m >> ((v + 1) % n) & 1) for v in range(n)):
            out.append(m)
    return out


@pytest.mark.parametrize("n", range(3, 13))
def test_zustandsraum_gleich_aufzaehlung_und_lucas(n):
    assert list(K.ring_states(n)) == brute_states(n)
    assert len(brute_states(n)) == K.lucas(n)


def dense_heatbath(n, lam):
    S_ = brute_states(n)
    ix = {m: i for i, m in enumerate(S_)}
    M = np.zeros((len(S_), len(S_)))
    p = lam / (1 + lam)
    for m in S_:
        for v in range(n):
            for heads, pc in ((True, p), (False, 1 - p)):
                nbs = (1 << ((v + 1) % n)) | (1 << ((v - 1) % n))
                new = (m | 1 << v) if heads and not m & nbs else m & ~(1 << v)
                M[ix[m], ix[new]] += pc / n
    return M, S_


def test_heatbath_matrix_gleich_erzeuger_matrix():
    for n in (4, 5, 6, 7):
        for lam in LAMS:
            M, _ = dense_heatbath(n, lam)
            assert np.allclose(K.get_ring(n).matrix("heatbath", lam).toarray(), M, atol=1e-15)


def test_metropolis_matrix_gleich_i_plus_q_durch_n_max():
    """Formel P = I + Q/(n·max(1, λ)) mit der Ratenmatrix Q (belegen mit Rate λ, leeren mit Rate 1), unabhängig aus den Zuständen aufgebaut."""
    for n in (4, 5, 6):
        for lam in LAMS + (10.0,):
            S_ = brute_states(n)
            ix = {m: i for i, m in enumerate(S_)}
            Q = np.zeros((len(S_), len(S_)))
            for a in S_:
                for b in S_:
                    diff = a ^ b
                    if diff and diff & (diff - 1) == 0:
                        Q[ix[a], ix[b]] = lam if b & diff else 1.0
                Q[ix[a], ix[a]] = -Q[ix[a]].sum()
            P_ = np.eye(len(S_)) + Q / (n * max(1.0, lam))
            assert np.allclose(K.get_ring(n).matrix("metropolis", lam).toarray(), P_, atol=1e-14)


def test_gleichgewicht_und_detailed_balance_beider_ketten():
    ring = K.get_ring(6)
    for lam in LAMS:
        pi = ring.pi(lam)
        w = lam ** ring.k.astype(float)
        assert np.allclose(pi, w / w.sum())
        for kind in ("heatbath", "metropolis"):
            Pm = ring.matrix(kind, lam).toarray()
            assert np.allclose(pi @ Pm, pi, atol=1e-14)
            assert np.allclose(pi[:, None] * Pm, (pi[:, None] * Pm).T, atol=1e-14)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Paarkette gegen Vorwärtsrechnung aus den Zufallsquellen (eigener Rechenweg, Wörterbuch statt Matrix)
# ---------------------------------------------------------------------------------------------------------------------------------------

def hb(x, n, v, heads):
    nbs = (1 << ((v + 1) % n)) | (1 << ((v - 1) % n))
    return (x | 1 << v) if heads and not x & nbs else x & ~(1 << v)


def metro(x, n, v, u, lam):
    pa, pr = min(1.0, lam), min(1.0, 1.0 / lam)
    nbs = (1 << ((v + 1) % n)) | (1 << ((v - 1) % n))
    if x >> v & 1:
        return x & ~(1 << v) if u < pr else x
    return (x | 1 << v) if not x & nbs and u < pa else x


def moves(coupling, n, lam):
    """Liste (Wahrscheinlichkeit, f(x, y)) der Zufallsquellen je Schritt, unabhängig von kpl_chain.kernels aufgeschrieben."""
    p = lam / (1 + lam)
    out = []
    if coupling == "S":
        for v in range(n):
            for h, pr in ((True, p), (False, 1 - p)):
                out.append((pr / n, lambda x, y, v=v, h=h: (hb(x, n, v, h), hb(y, n, v, h))))
    elif coupling == "C":
        for v in range(n):
            for hx, px in ((True, p), (False, 1 - p)):
                for hy, py in ((True, p), (False, 1 - p)):
                    out.append((px * py / n, lambda x, y, v=v, hx=hx, hy=hy: (hb(x, n, v, hx), hb(y, n, v, hy))))
    elif coupling == "I":
        for v in range(n):
            for hx, px in ((True, p), (False, 1 - p)):
                for w in range(n):
                    for hy, py in ((True, p), (False, 1 - p)):
                        out.append((px * py / (n * n), lambda x, y, v=v, hx=hx, w=w, hy=hy: (hb(x, n, v, hx), hb(y, n, w, hy))))
    elif coupling == "M":
        pa, pr = min(1.0, lam), min(1.0, 1.0 / lam)
        cuts = sorted({0.0, pa, pr, 1.0})
        for v in range(n):
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                if hi > lo:
                    u = 0.5 * (lo + hi)
                    out.append(((hi - lo) / n, lambda x, y, v=v, u=u: (metro(x, n, v, u, lam), metro(y, n, v, u, lam))))
    return out


def forward_survival(coupling, n, lam, start, T):
    mv = moves(coupling, n, lam)
    dist = {start: 1.0}
    out = [1.0]
    for _ in range(T):
        nd = {}
        for (x, y), pr in dist.items():
            for q, f in mv:
                a, b = f(x, y)
                if a != b:
                    nd[(a, b)] = nd.get((a, b), 0.0) + pr * q
        dist = nd
        out.append(sum(dist.values()))
    return np.array(out)


# die Wörterbuch-Rechnung der unabhängigen Kopplung (n² Platzpaare je Schritt) gibt es nur für n = 4
FAELLE = [(c, n, lam) for c in ("S", "C", "I", "M") for (n, lam) in [(4, 2.0), (4, 1.0), (6, 5.0)] if not (c == "I" and n > 4)]


@pytest.mark.parametrize("coupling,n,lam", FAELLE)
def test_paarkette_gleich_vorwaertsrechnung_aus_den_zufallsquellen(coupling, n, lam):
    ring = K.get_ring(n)
    top, bot = K.peak_masks(n)
    pc = P.PairChain(ring, coupling, lam)
    st = pc.pair(ring.index(top), ring.index(bot))
    T = 25
    ex, _ = pc.survival(st, t_stop=0.0, t_cap=T)
    ref = forward_survival(coupling, n, lam, (top, bot), T)
    assert np.allclose(ex[: T + 1], ref, atol=1e-12)


def test_ausspielen_aller_folgen_n6_lambda2():
    """P(τ > t) für t = 1 … 4 durch Ausspielen aller (Platz, Münze)^t-Folgen (12^4 = 20 736 Folgen) gegen die Paarkette."""
    n, lam = 6, 2.0
    ring = K.get_ring(n)
    top, bot = K.peak_masks(n)
    pc = P.PairChain(ring, "S", lam)
    ex, _ = pc.survival(pc.pair(ring.index(top), ring.index(bot)), t_stop=0.0, t_cap=4)
    p = lam / (1 + lam)
    mv = [(v, h) for v in range(n) for h in (True, False)]
    w = {m: (p if m[1] else 1 - p) / n for m in mv}
    for T in range(1, 5):
        terms = []
        for seq in itertools.product(mv, repeat=T):
            a, b, pr = top, bot, 1.0
            for m in seq:
                a, b = hb(a, n, *m), hb(b, n, *m)
                pr *= w[m]
            if a != b:
                terms.append(pr)
        assert math.isclose(math.fsum(terms), ex[T], abs_tol=1e-12), T


def test_randketten_der_paarkerne_sind_die_kette():
    """Jede der Kopplungen S, C, M ist eine Kopplung: Randverteilung von x und von y gleich der Übergangsmatrix (Abweichung < 1e-15)."""
    worst = 0.0
    for n in (4, 6):
        ring = K.get_ring(n)
        for lam in (0.5, 1.0, 5.0):
            for c, kind in (("S", "heatbath"), ("C", "heatbath"), ("M", "metropolis")):
                Mx = np.zeros((ring.N, ring.N))
                My = np.zeros((ring.N, ring.N))
                ar = np.arange(ring.N)
                for pr, fx, fy in K.kernels(ring, c, lam):
                    np.add.at(Mx, (ar, fx), pr)
                    np.add.at(My, (ar, fy), pr)
                Pm = ring.matrix(kind, lam).toarray()
                worst = max(worst, np.abs(Mx - Pm).max(), np.abs(My - Pm).max())
    assert worst < 1e-15


# ---------------------------------------------------------------------------------------------------------------------------------------
# Abstände, Mittel
# ---------------------------------------------------------------------------------------------------------------------------------------

def test_d_und_dbar_aus_spektralzerlegung_gleich_matrixpotenz():
    for n, lam in ((4, 1.0), (4, 5.0), (6, 2.0), (6, 0.5)):
        ring = K.get_ring(n)
        pi = ring.pi(lam)
        Pm = ring.matrix("heatbath", lam).toarray()
        spec = E.Spectrum(Pm, pi)
        for t in (1, 2, 7, 30):
            Pt = np.linalg.matrix_power(Pm, t)
            d = 0.5 * np.abs(Pt - pi).sum(1).max()
            db = max(0.5 * np.abs(Pt[i] - Pt[j]).sum() for i in range(ring.N) for j in range(i))
            d2, db2 = spec.d_dbar(t)
            assert math.isclose(d, d2, abs_tol=1e-12) and math.isclose(db, db2, abs_tol=1e-12), (n, lam, t)


def test_mischzeit_gleich_brute_force_ueber_alle_starts():
    for n, lam in ((4, 1.0), (6, 1.0), (6, 5.0)):
        ring = K.get_ring(n)
        pi = ring.pi(lam)
        Pm = ring.matrix("heatbath", lam).toarray()
        Pt = np.eye(ring.N)
        t = 0
        while True:
            t += 1
            Pt = Pt @ Pm
            if 0.5 * np.abs(Pt - pi).sum(1).max() <= 0.25:
                break
        assert E.mixing_times(E.Spectrum(Pm, pi))[0] == t


def test_erwartung_gleich_reihe_der_ueberlebenswahrscheinlichkeiten():
    for coupling in ("S", "C", "M"):
        ring = K.get_ring(6)
        pc = P.PairChain(ring, coupling, 2.0)
        top, bot = K.peak_masks(6)
        st = pc.pair(ring.index(top), ring.index(bot))
        ex, _ = pc.survival(st, t_stop=1e-15, t_cap=50_000)
        assert math.isclose(ex[:-1].sum(), pc.mean_times(st), rel_tol=1e-8), coupling


def test_hyperwuerfel_zaehlkette_gleich_sammlerformel_und_paarkette():
    for m in (3, 4, 6):
        formel = np.array([1 - sum((-1) ** j * math.comb(m, j) * (1 - j / m) ** t for j in range(m + 1)) for t in range(61)])
        assert np.allclose(E.collector_survival(m, 60), formel, atol=1e-13)
        # Paarkette aus den Zufallsquellen (Koordinate, Bit): ein dritter Rechenweg
        fs = [(1.0 / (2 * m), v, b) for v in range(m) for b in (0, 1)]
        dist = {(0, (1 << m) - 1): 1.0}
        for t in range(61):
            assert abs(sum(dist.values()) - formel[t]) < 1e-12, (m, t)
            nd = {}
            for (x, y), pr in dist.items():
                for q, v, b in fs:
                    a2, b2 = (x & ~(1 << v)) | (b << v), (y & ~(1 << v)) | (b << v)
                    if a2 != b2:
                        nd[(a2, b2)] = nd.get((a2, b2), 0.0) + pr * q
            dist = nd


def test_hyperwuerfel_mischzeit_gleich_aufzaehlung_aller_starts():
    """d(t) mit allen 2^m Starts: der Start alle 0 ist der schlechteste (Symmetrie), t_mix und t̄_mix der Zählkette gleich der vollen Rechnung."""
    for m in (4, 6):
        N = 1 << m
        Pm = np.zeros((N, N))
        for x in range(N):
            for v in range(m):
                for b in (0, 1):
                    Pm[x, (x & ~(1 << v)) | (b << v)] += 1 / (2 * m)
        pi = np.full(N, 1.0 / N)
        Pt = np.eye(N)
        t_mix = t_bar = None
        t = 0
        while t_mix is None or t_bar is None:
            t += 1
            Pt = Pt @ Pm
            if t_mix is None and 0.5 * np.abs(Pt - pi).sum(1).max() <= 0.25:
                t_mix = t
            if t_bar is None and max(0.5 * np.abs(Pt[i] - Pt[j]).sum() for i in range(N) for j in range(i)) <= 0.25:
                t_bar = t
        row = E.hypercube_row(m)
        assert (row["t_mix"], row["t_bar"]) == (t_mix, t_bar)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Simulation (SplitMix64) gegen die exakte Verteilung
# ---------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("coupling,n,lam,R", [("S", 6, 2.0, 3000), ("C", 6, 2.0, 2000), ("I", 4, 2.0, 3000), ("M", 6, 5.0, 2000), ("S", 8, 5.0, 1500)])
def test_simulierte_treffzeiten_gleich_exakter_verteilung(coupling, n, lam, R):
    ring = K.get_ring(n)
    top, bot = K.peak_masks(n)
    pc = P.PairChain(ring, coupling, lam)
    st = pc.pair(ring.index(top), ring.index(bot))
    ex, _ = pc.survival(st, t_stop=1e-6, t_cap=200_000)
    mean = pc.mean_times(st)
    taus, cens, _ = S.sample_taus(n, lam, coupling, R, 4242 + n, top, bot, 10_000_000)
    assert cens == 0
    a = np.array(taus, dtype=float)
    z = (a.mean() - mean) / (a.std(ddof=1) / math.sqrt(R))
    assert abs(z) < 4, z
    ts = np.arange(len(ex))
    emp = np.array([(a > t).mean() for t in ts])
    assert np.abs(emp - ex).max() <= 2 * 1.36 / math.sqrt(R)


# ---------------------------------------------------------------------------------------------------------------------------------------
# Ordnungsstatistik des Stichprobenquantils
# ---------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("R", [1, 2, 3, 4, 5, 6, 7])
def test_stichprobenquantil_verteilung_gleich_aufzaehlung(R):
    pmf = {1: 0.1, 2: 0.25, 3: 0.3, 4: 0.2, 5: 0.15}
    surv = np.array([1.0] + [1.0 - sum(v for k, v in pmf.items() if k <= t) for t in range(1, 6)])
    cdf = E.quantile_estimate_cdf(surv, R)
    brute = np.zeros(6)
    for outcome in itertools.product(pmf, repeat=R):
        pr = math.prod(pmf[x] for x in outcome)
        q = sorted(outcome)[E.sample_rank(R) - 1]
        brute[q:] += pr
    assert np.allclose(cdf, brute, atol=1e-12)


def test_rang_des_stichprobenquantils():
    assert [E.sample_rank(R) for R in (1, 2, 3, 4, 5, 10, 20, 50, 200, 1000)] == [1, 2, 3, 3, 4, 8, 15, 38, 150, 750]
