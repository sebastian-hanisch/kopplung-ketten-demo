"""Gekoppelte Läufe: jeder Schritt der Läufe gleich der Aktualisierungsfunktion der exakten Rechnung, Reproduzierbarkeit, Budget, Eingaben."""

import pytest

import kpl_chain as K
import kpl_sim as S
from kpl_rng import SplitMix64


class Fake:
    """Zufallsquelle mit vorgegebenen Werten: `below` und `random` geben die Listen der Reihe nach zurück."""

    def __init__(self, below, rnd):
        self._b, self._r = list(below), list(rnd)

    def below(self, n):
        return self._b.pop(0)

    def random(self):
        return self._r.pop(0)


HEADS, TAILS = 0.0, 0.999            # u < λ/(1+λ) gilt für 0,0 (Kopf), nicht für 0,999 (Zahl), solange λ/(1+λ) < 0,999


@pytest.mark.parametrize("n", [4, 5, 6])
def test_schritt_s_gleich_aktualisierungsfunktion(n):
    ring = K.get_ring(n)
    lam = 2.0
    for v in range(n):
        for heads, u in ((True, HEADS), (False, TAILS)):
            f = ring.phi_heatbath(v, heads)
            for i, x in enumerate(ring.S):
                for j, y in enumerate(ring.S):
                    step = S.make_step(Fake([v], [u]), n, lam, "S")
                    assert step(int(x), int(y)) == (int(ring.S[f[i]]), int(ring.S[f[j]]))


@pytest.mark.parametrize("n", [4, 5])
def test_schritt_c_i_gleich_aktualisierungsfunktion(n):
    ring = K.get_ring(n)
    lam = 2.0
    for v in range(n):
        for hx, ux in ((True, HEADS), (False, TAILS)):
            for hy, uy in ((True, HEADS), (False, TAILS)):
                fx, fy = ring.phi_heatbath(v, hx), ring.phi_heatbath(v, hy)
                for i, x in enumerate(ring.S):
                    for j, y in enumerate(ring.S):
                        step = S.make_step(Fake([v], [ux, uy]), n, lam, "C")
                        assert step(int(x), int(y)) == (int(ring.S[fx[i]]), int(ring.S[fy[j]]))
                w = (v + 1) % n
                fw = ring.phi_heatbath(w, hy)
                for i, x in enumerate(ring.S):
                    for j, y in enumerate(ring.S):
                        step = S.make_step(Fake([v, w], [ux, uy]), n, lam, "I")
                        assert step(int(x), int(y)) == (int(ring.S[fx[i]]), int(ring.S[fw[j]]))


@pytest.mark.parametrize("lam", [0.5, 1.0, 5.0])
def test_schritt_m_gleich_aktualisierungsfunktion(lam):
    n = 5
    ring = K.get_ring(n)
    pa, pr = min(1.0, lam), min(1.0, 1.0 / lam)
    cuts = sorted({0.0, pa, pr, 1.0})
    for v in range(n):
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if hi <= lo:
                continue
            u = 0.5 * (lo + hi)
            f = ring.phi_metropolis(v, u, lam)
            for i, x in enumerate(ring.S):
                for j, y in enumerate(ring.S):
                    step = S.make_step(Fake([v], [u]), n, lam, "M")
                    assert step(int(x), int(y)) == (int(ring.S[f[i]]), int(ring.S[f[j]]))


def test_lauf_ist_reproduzierbar_und_unabhaengig_von_der_laeufezahl():
    top, bot = K.peak_masks(6)
    a, _, _ = S.sample_taus(6, 2.0, "S", 10, 77, top, bot, 10**6)
    b, _, _ = S.sample_taus(6, 2.0, "S", 25, 77, top, bot, 10**6)
    c, _, _ = S.sample_taus(6, 2.0, "S", 10, 78, top, bot, 10**6)
    assert b[:10] == a and c != a


def test_lauf_aufzeichnung_und_invarianten():
    n, lam = 8, 5.0
    top, bot = K.peak_masks(n)
    for coupling in S.COUPLINGS:
        r = S.run_pair(SplitMix64(5), n, lam, coupling, top, bot, 5000, record=True)
        assert r.xs[0] == top and r.ys[0] == bot and len(r.xs) == r.steps + 1
        assert all(K.is_independent(x, n) for x in r.xs) and all(K.is_independent(y, n) for y in r.ys)
        for a, b in zip(r.xs, r.xs[1:]):
            assert bin(a ^ b).count("1") <= 1                    # ein Schritt ändert höchstens einen Platz je Kette
        for a, b in zip(r.ys, r.ys[1:]):
            assert bin(a ^ b).count("1") <= 1
        if r.tau is not None:
            assert r.xs[-1] == r.ys[-1] and all(x != y for x, y in zip(r.xs[:-1], r.ys[:-1]))


def test_gleiches_startpaar_gibt_tau_null_und_obergrenze_gibt_none():
    r = S.run_pair(SplitMix64(1), 6, 1.0, "S", 5, 5, 100)
    assert r.tau == 0 and r.steps == 0
    top, bot = K.peak_masks(6)
    r = S.run_pair(SplitMix64(1), 6, 1.0, "S", top, bot, 3)
    assert r.tau is None and r.steps == 3                         # τ ≥ n = 6: in 3 Schritten treffen sich die Gipfel nicht


def test_laeufebudget_wird_vor_dem_rechnen_festgelegt():
    assert S.run_budget(1000, 265.3, 300_000) == 1000
    assert S.run_budget(1000, 2548.1, 300_000) == 117
    assert S.run_budget(10, 1e9, 300_000) == 1                    # mindestens ein Lauf
    assert S.run_budget(1, 10.0, 300_000) == 1


def test_eingabe_einer_belegung():
    assert S.parse_state("10101010", 8) == (0b01010101, "")
    assert S.parse_state(" 1010 1010 ", 8)[0] == 0b01010101
    for bad in ("1010", "1010101a", "", "11000000", "10000001"):      # falsche Länge, falsches Zeichen, leer, Nachbarn belegt, Platz 7 und Platz 0 belegt (Ring geschlossen)
        mask, msg = S.parse_state(bad, 8)
        assert mask is None and msg
    for m in (0, 1, 0b0101, 0b1010):
        assert S.parse_state(S.mask_text(m, 4), 4)[0] == m


def test_zufaellige_belegungen():
    for n in (4, 6, 9, 24):
        x, y = S.random_pair(11, n)
        assert x != y and K.is_independent(x, n) and K.is_independent(y, n)
        assert S.random_pair(11, n) == (x, y)
