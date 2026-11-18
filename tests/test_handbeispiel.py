"""Handbeispiel in exakten Brüchen: Ring mit n = 4 Plätzen, λ = 1 (7 Zustände), Heat-bath, Kopplung S, Gipfelpaar ({0, 2} gegen {1, 3}).

Jede Zahl ist von Hand nachgerechnet (Zeilen im Kommentar) und mit Brüchen (`fractions`) gegen einen eigenen, von der Demo unabhängigen Rechenweg geprüft; danach stehen die Zahlen
der Demo (numpy, Gleitkomma) gegen dieselben Brüche. Kein Zahlenwert ist aus dem Code der Demo abgeschrieben."""

import itertools
import math
from fractions import Fraction as F

import numpy as np

import kpl_chain as K
import kpl_exact as E
import kpl_pairs as P

n = 4
FULL = (1 << n) - 1


def rot(m):
    return ((m << 1) | (m >> (n - 1))) & FULL


STATES = [m for m in range(1 << n) if m & rot(m) == 0]
IDX = {m: i for i, m in enumerate(STATES)}
NS = len(STATES)
EVEN, ODD = 0b0101, 0b1010


def nb(v):
    return (1 << ((v + 1) % n)) | (1 << ((v - 1) % n))


def upd(x, v, heads):
    bit = 1 << v
    return x | bit if heads and (x & nb(v)) == 0 else x & ~bit


MOVES = [(v, h) for v in range(n) for h in (True, False)]
W = F(1, 8)          # Platz 1/4 mal Münze 1/2 (λ = 1)


def build_P():
    M = [[F(0)] * NS for _ in range(NS)]
    for x in STATES:
        for v, h in MOVES:
            M[IDX[x]][IDX[upd(x, v, h)]] += W
    return M


def matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(NS) if A[i][k]) for j in range(NS)] for i in range(NS)]


def tv(p, q):
    return sum(abs(a - b) for a, b in zip(p, q)) / 2


PM = build_P()
PI = [F(1, 7)] * NS


def powers(T):
    Pt = [[F(int(i == j)) for j in range(NS)] for i in range(NS)]
    d, db = {}, {}
    for t in range(1, T + 1):
        Pt = matmul(Pt, PM)
        d[t] = max(tv(Pt[i], PI) for i in range(NS))
        db[t] = max(tv(Pt[i], Pt[j]) for i in range(NS) for j in range(i + 1, NS))
    return d, db


D, DB = powers(40)


def step_pair(x, y, mv):
    return upd(x, mv[0], mv[1]), upd(y, mv[0], mv[1])


def forward_tail(T):
    dist = {(EVEN, ODD): F(1)}
    out = [F(1)]
    for _ in range(T):
        nd = {}
        for (x, y), p in dist.items():
            for mv in MOVES:
                a, b = step_pair(x, y, mv)
                if a != b:
                    nd[(a, b)] = nd.get((a, b), 0) + p * W
        dist = nd
        out.append(sum(dist.values()))
    return out


TAIL = forward_tail(45)


def test_sieben_zustaende_und_uebergangsmatrix_von_hand():
    assert NS == 7
    assert sorted(STATES) == sorted([0, 1, 2, 4, 8, EVEN, ODD])
    assert all(sum(row) == 1 for row in PM)
    # leer: Münze Zahl (1/2) oder Kopf auf einem von vier Plätzen (je 1/8): bleibt leer mit 4/8 = 1/2
    assert PM[IDX[0]][IDX[0]] == F(1, 2) and all(PM[IDX[0]][IDX[1 << v]] == F(1, 8) for v in range(n))
    # {0}: Platz 0 Kopf bleibt, Platz 0 Zahl leert (1/8), Plätze 1 und 3 Kopf: Nachbar 0 belegt, also leer gesetzt (bleibt), Platz 2 Kopf belegt (1/8), Platz 2 Zahl bleibt
    assert PM[IDX[1]][IDX[1]] == F(3, 4) and PM[IDX[1]][IDX[0]] == F(1, 8) and PM[IDX[1]][IDX[EVEN]] == F(1, 8)
    assert PM[IDX[EVEN]][IDX[EVEN]] == F(3, 4) and PM[IDX[EVEN]][IDX[1]] == F(1, 8) and PM[IDX[EVEN]][IDX[4]] == F(1, 8)
    # Gleichgewicht: 1/7 je Zustand (λ = 1), Detailed Balance
    assert [sum(PI[i] * PM[i][j] for i in range(NS)) for j in range(NS)] == PI
    assert all(PI[i] * PM[i][j] == PI[j] * PM[j][i] for i in range(NS) for j in range(NS))


def test_die_matrix_der_demo_gleicht_den_bruechen():
    ring = K.get_ring(4)
    Pn = ring.matrix("heatbath", 1.0).toarray()
    for x in STATES:
        for y in STATES:
            assert math.isclose(Pn[ring.index(x), ring.index(y)], float(PM[IDX[x]][IDX[y]]), abs_tol=1e-15)
    assert np.allclose(ring.pi(1.0), 1 / 7)


def test_abstaende_und_mischzeiten_beide_zaehleinheiten():
    assert D[1] == F(17, 28)          # Zeile von {0,2}: 3/4 auf {0,2}, je 1/8 auf {0} und {2}; gegen 1/7 überschreitet nur 3/4 − 1/7 = 17/28
    assert DB[1] == 1                 # nach einem Schritt haben die Gipfel disjunkte Träger
    assert DB[2] == F(31, 32)
    t_mix = min(t for t in D if D[t] <= F(1, 4))
    t_bar = min(t for t in DB if DB[t] <= F(1, 4))
    t_pw = min(t for t in DB if float(DB[t]) <= 1 / math.e)
    assert (t_mix, t_bar, t_pw) == (12, 21, 16)
    assert D[11] > F(1, 4) >= D[12] and DB[20] > F(1, 4) >= DB[21]
    assert all(D[t] <= DB[t] <= 2 * D[t] for t in D)
    spec = E.Spectrum(K.get_ring(4).matrix("heatbath", 1.0), K.get_ring(4).pi(1.0))
    assert E.mixing_times(spec) == (12, 21, 16)
    for t in (1, 2, 5, 12, 21, 40):
        d_n, db_n = spec.d_dbar(t)
        assert math.isclose(d_n, float(D[t]), abs_tol=1e-12) and math.isclose(db_n, float(DB[t]), abs_tol=1e-12)


def test_ueberlebenswahrscheinlichkeit_der_kopplung_von_hand():
    assert TAIL[0] == TAIL[1] == TAIL[2] == TAIL[3] == 1       # die Zustände unterscheiden sich an 4 Plätzen, ein Schritt ändert höchstens einen: τ ≥ 4
    assert TAIL[4] == F(63, 64)                                # genau 64 von 4096 Folgen aus (Platz, Münze) treffen nach 4 Schritten: P(τ = 4) = 1/64
    assert TAIL[5] == F(977, 1024) and TAIL[6] == F(1871, 2048)


def test_ausspielen_aller_folgen_gleicht_der_paarkette():
    for t in range(1, 6):
        zaehler = 0
        for seq in itertools.product(MOVES, repeat=t):
            a, b = EVEN, ODD
            for mv in seq:
                a, b = step_pair(a, b, mv)
            zaehler += a != b
        assert F(zaehler, 8 ** t) == TAIL[t], t


def test_die_paarkette_der_demo_gleicht_den_bruechen():
    ring = K.get_ring(4)
    pc = P.PairChain(ring, "S", 1.0)
    st = pc.pair(ring.index(ring.top), ring.index(ring.bot))
    ex, mx = pc.survival(st, t_stop=1e-12)
    for t in range(0, 46):
        assert math.isclose(ex[t], float(TAIL[t]), abs_tol=1e-13), t
        assert math.isclose(mx[t], float(TAIL[t]), abs_tol=1e-13), t    # das Maximum über alle 42 Paare ist das Gipfelpaar
    assert pc.n_pairs == 42 and int(pc.reachable(st).sum()) == 19


def solve_mean(start):
    seen, stack = {start}, [start]
    while stack:
        x, y = stack.pop()
        for mv in MOVES:
            a, b = step_pair(x, y, mv)
            if a != b and (a, b) not in seen:
                seen.add((a, b))
                stack.append((a, b))
    pairs = sorted(seen)
    ix = {p: i for i, p in enumerate(pairs)}
    m = len(pairs)
    A = [[F(0)] * (m + 1) for _ in range(m)]
    for (x, y) in pairs:
        i = ix[(x, y)]
        A[i][i] += 1
        A[i][m] = F(1)
        for mv in MOVES:
            a, b = step_pair(x, y, mv)
            if a != b:
                A[i][ix[(a, b)]] -= W
    for c in range(m):
        piv = next(r for r in range(c, m) if A[r][c] != 0)
        A[c], A[piv] = A[piv], A[c]
        pv = A[c][c]
        A[c] = [v / pv for v in A[c]]
        for r in range(m):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return {p: A[ix[p]][m] for p in pairs}


def test_erwartung_und_schranken():
    means = solve_mean((EVEN, ODD))
    assert len(means) == 19
    assert means[(EVEN, ODD)] == F(135, 7) and max(means.values()) == F(135, 7)
    # E[τ] = Σ_t P(τ > t): die Reihe bis 4 000 Schritte stimmt auf 1e-9 (Gleitkomma-Gegenprobe)
    ring = K.get_ring(4)
    pc = P.PairChain(ring, "S", 1.0)
    st = pc.pair(ring.index(EVEN), ring.index(ODD))
    ex, _ = pc.survival(st, t_stop=1e-14, t_cap=4000)
    assert abs(ex[:-1].sum() - 135 / 7) < 1e-9
    assert math.isclose(pc.mean_times(st), 135 / 7, rel_tol=1e-12)
    t_c = min(t for t in range(len(TAIL)) if TAIL[t] <= F(1, 4))
    assert t_c == 25 and TAIL[24] > F(1, 4)
    assert 12 <= 21 <= t_c                                # t_mix ≤ t̄_mix ≤ t_c: die Kopplung ist eine obere Schranke
    assert 4 * F(135, 7) == F(540, 7) and F(540, 7) > 77 and F(540, 7) < 78     # LPW 5.5: t_mix ≤ 4 E[τ] = 77,1
    assert P.tail_quantile(ex, 0.25) == 25


def test_ein_lauf_von_hand():
    lauf = [(1, False), (3, False), (0, True), (2, True)]
    x, y = EVEN, ODD
    spur = [(x, y)]
    for mv in lauf:
        x, y = step_pair(x, y, mv)
        spur.append((x, y))
    assert spur == [(5, 10), (5, 8), (5, 0), (5, 1), (5, 5)]        # unten: {1,3} → {3} → leer → {0} → {0,2}; Treffen bei t = 4 in {0, 2}
    x, y = EVEN, ODD
    for mv in [(0, True), (1, True), (2, True), (3, True)]:
        x, y = step_pair(x, y, mv)
    assert (x, y) == (EVEN, ODD)                                    # vier Mal Kopf: nichts ändert sich
    x, y = EVEN, ODD
    for mv in [(0, False), (1, False), (2, False), (3, False)]:
        x, y = step_pair(x, y, mv)
    assert x == y == 0                                              # vier Mal Zahl auf vier Plätzen: beide leer, Treffen bei t = 4


def test_satz_5_nach_quellenbericht_mit_ell_4():
    for t in range(1, 41):
        assert TAIL[t] / 4 <= DB[t] <= TAIL[t], t
        assert D[t] <= TAIL[t], t
    assert all(TAIL[t] / 4 <= D[t] for t in range(1, 41))           # eine Eigenheit von n = 4: P/ℓ bleibt auch unter d
    ring = K.get_ring(4)
    assert E.longest_chain(ring) == 4


def test_drei_zustaende_d_gegen_dbar():
    P3 = [[F(1, 2), F(1, 2), F(0)], [F(1, 2), F(0), F(1, 2)], [F(0), F(1, 2), F(1, 2)]]
    pi3 = [F(1, 3)] * 3
    assert [sum(pi3[i] * P3[i][j] for i in range(3)) for j in range(3)] == pi3
    Pt = [[F(int(i == j)) for j in range(3)] for i in range(3)]
    d, db = {}, {}
    for k in range(1, 11):
        Pt = [[sum(Pt[i][m] * P3[m][j] for m in range(3)) for j in range(3)] for i in range(3)]
        d[k] = max(sum(abs(Pt[i][j] - pi3[j]) for j in range(3)) / 2 for i in range(3))
        db[k] = max(sum(abs(Pt[i][j] - Pt[i2][j]) for j in range(3)) / 2 for i in range(3) for i2 in range(i + 1, 3))
    assert d[1] == F(1, 3) and all(db[k] == F(1, 2 ** k) for k in range(1, 11))
    # Kopplung mit gleicher Münze, oben 2 unten 0; die Treffzeit ist die Zeit, bis eine Folge von Zahl (oder Kopf) beide an den Rand gebracht hat
    tail = {0: F(1), 1: F(1), 2: F(1, 2), 3: F(1, 4), 4: F(1, 8)}
    ts = E.three_state_example(10)
    for k in range(1, 5):
        assert ts["rows"][k - 1]["tail"] == tail[k]
    for r in ts["rows"]:
        assert r["tail_over_ell"] == db[r["k"]] and r["d"] == d[r["k"]] and r["dbar"] == db[r["k"]]
    assert ts["rows"][0]["tail_over_ell"] == F(1, 2) > d[1] == F(1, 3)      # 1/2 > 1/3: gegen d geprüft sähe Satz 5 verletzt aus
    assert ts["E_tau"] == 3


def test_hyperwuerfel_sammlerformel_von_hand():
    m = 3
    for t in range(16):
        formel = 1 - sum((-1) ** j * math.comb(m, j) * F(m - j, m) ** t for j in range(m + 1))
        assert math.isclose(E.collector_survival(m, 15)[t], float(formel), abs_tol=1e-14), t
    assert sum((-1) ** (j + 1) * math.comb(m, j) * F(m, j) for j in range(1, m + 1)) == F(11, 2) == 3 * (1 + F(1, 2) + F(1, 3))
    assert math.isclose(E.hypercube_row(3)["E_tau"], 5.5)
