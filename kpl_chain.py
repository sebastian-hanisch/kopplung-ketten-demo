"""Der Hard-Core-Ring als Markov-Kette: Zustände, Übergangsmatrizen, Aktualisierungsfunktionen und Aktualisierungsschritte für einzelne Zustände.

Modell (ausgedacht, keine reale Datenquelle): n Stellplätze im Kreis (Plätze 0 bis n−1), ein Zustand ist die Menge der belegten Plätze, keine zwei
benachbarten Plätze sind belegt, das Gewicht eines Zustands mit k belegten Plätzen ist λ^k (Levin/Peres/Wilmer, Abschnitt 3.3.4). Die Zahl der Zustände ist die
Lucas-Zahl L_n (n = 4: 7, n = 8: 47, n = 10: 123). Zustände sind Bitmasken (Bit v gesetzt heißt: Platz v belegt).

Zwei Ketten, die dieselbe Verteilung als Gleichgewicht haben:
* **Heat-bath** (Gibbs-Sampler, Glauber): ein Platz gleichverteilt, Münze mit Kopf-Wahrscheinlichkeit λ/(1+λ); Kopf belegt den Platz, wenn beide Nachbarn leer
  sind, sonst wird der Platz leer gesetzt.
* **Metropolis** (Einzelplatz): ein Platz gleichverteilt, Umkehren (belegen oder leeren), angenommen mit Wahrscheinlichkeit min(1, π(neu)/π(alt)).

Eigene Neuschreibung (kein Import aus anderen Demos); die Übergangsmatrix des Heat-bath-Rings kommt in den Nachbar-Demos der Reihe ebenfalls vor, dort mit eigenem Code."""

from functools import lru_cache

import numpy as np
import scipy.sparse as sp


def peak_masks(n: int) -> tuple[int, int]:
    """Gipfelpaar: alle geraden Plätze belegt gegen alle ungeraden Plätze belegt (größtes und kleinstes Element der Ordnung nach Levin/Peres/Wilmer, Beispiel 22.10).
    Bei ungeradem n sind Platz 0 und Platz n−1 Nachbarn: dann sind es die maximalen Belegungen auf den Plätzen 0 bis n−2 (gerade bzw. ungerade Plätze), die Ordnung fehlt."""
    if n < 4:
        raise ValueError("Der Ring braucht mindestens 4 Plätze")
    top = sum(1 << v for v in range(0, n - (n % 2), 2))
    bot = sum(1 << v for v in range(1, n - (n % 2), 2))
    return top, bot


def popcount(a):
    a = np.asarray(a, dtype=np.int64)
    c = np.zeros(a.shape, dtype=np.int64)
    b = a.copy()
    while b.any():
        c += b & 1
        b >>= 1
    return c


def neighbour_mask(n: int, v: int) -> int:
    return (1 << ((v + 1) % n)) | (1 << ((v - 1) % n))


def is_independent(mask: int, n: int) -> bool:
    """Keine zwei benachbarten Plätze belegt (der Ring ist geschlossen: Platz n−1 grenzt an Platz 0)."""
    rot = ((mask << 1) | (mask >> (n - 1))) & ((1 << n) - 1)
    return mask & rot == 0


@lru_cache(maxsize=None)
def ring_states(n: int):
    """Alle zulässigen Zustände des n-Rings, aufsteigend sortiert (Erzeugung ohne die 2^n Masken aufzuzählen: Wege ohne Nachbarn, dann den Ring schließen)."""
    if n < 3:
        raise ValueError("Der Ring braucht mindestens 3 Plätze")
    arr = np.zeros(1, dtype=np.int64)
    for v in range(n):
        ok = (arr & (1 << (v - 1) if v else 0)) == 0
        arr = np.concatenate([arr, arr[ok] | (1 << v)])
    closed = ~(((arr & 1) != 0) & ((arr >> (n - 1)) & 1 != 0))
    out = np.sort(arr[closed])
    out.setflags(write=False)
    return out


class Ring:
    """Zustandsraum des n-Rings mit den Index-Abbildungen der Aktualisierungen. Zustände sind Indizes 0..N−1 in `S` (sortierte Bitmasken)."""

    def __init__(self, n: int):
        self.n = n
        self.S = ring_states(n)
        self.N = len(self.S)
        self.k = popcount(self.S)
        self.top, self.bot = peak_masks(n) if n >= 4 else (None, None)
        self.nbm = np.array([neighbour_mask(n, v) for v in range(n)], dtype=np.int64)
        ai, aj, ri, rj = [], [], [], []
        for v in range(n):
            bit = 1 << v
            occ = (self.S & bit) != 0
            free = (self.S & self.nbm[v]) == 0
            ia = np.where(~occ & free)[0]
            ai.append(ia)
            aj.append(np.searchsorted(self.S, self.S[ia] | bit))
            ir = np.where(occ)[0]
            ri.append(ir)
            rj.append(np.searchsorted(self.S, self.S[ir] & ~bit))
        self.I_add, self.J_add = np.concatenate(ai), np.concatenate(aj)
        self.I_rem, self.J_rem = np.concatenate(ri), np.concatenate(rj)

    def index(self, mask: int) -> int:
        i = int(np.searchsorted(self.S, mask))
        if i >= self.N or self.S[i] != mask:
            raise ValueError("Keine zulässige Belegung")
        return i

    def pi(self, lam: float):
        """Gleichgewicht π(x) ∝ λ^|x| (stabil gerechnet über den Logarithmus)."""
        e = self.k * np.log(lam)
        w = np.exp(e - e.max())
        return w / w.sum()

    def rates(self, kind: str, lam: float) -> tuple[float, float]:
        """(Wahrscheinlichkeit, einen freien Platz zu belegen, Wahrscheinlichkeit, einen belegten zu leeren) bei gewähltem Platz."""
        if kind == "heatbath":
            return lam / (1 + lam), 1 / (1 + lam)
        if kind == "metropolis":
            return min(1.0, lam), min(1.0, 1.0 / lam)
        raise ValueError(kind)

    def matrix(self, kind: str, lam: float):
        """Übergangsmatrix der Kette als dünn besetzte Matrix (N×N)."""
        pa, pr = self.rates(kind, lam)
        rows = np.concatenate([self.I_add, self.I_rem])
        cols = np.concatenate([self.J_add, self.J_rem])
        vals = np.concatenate([np.full(len(self.I_add), pa / self.n), np.full(len(self.I_rem), pr / self.n)])
        off = sp.csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))
        diag = 1.0 - np.asarray(off.sum(axis=1)).ravel()
        return (off + sp.diags(diag)).tocsr()

    # --- Aktualisierungsfunktionen als Index-Abbildungen N → N -------------------------------------------------------------------------------

    def phi_heatbath(self, v: int, heads: bool):
        bit = 1 << v
        if heads:
            free = (self.S & self.nbm[v]) == 0
            y = np.where(free, self.S | bit, self.S & ~bit)
        else:
            y = self.S & ~bit
        return np.searchsorted(self.S, y)

    def phi_metropolis(self, v: int, u: float, lam: float):
        pa, pr = self.rates("metropolis", lam)
        bit = 1 << v
        occ = (self.S & bit) != 0
        free = (self.S & self.nbm[v]) == 0
        add = (~occ) & free & (u < pa)
        rem = occ & (u < pr)
        y = np.where(add, self.S | bit, np.where(rem, self.S & ~bit, self.S))
        return np.searchsorted(self.S, y)


@lru_cache(maxsize=None)
def get_ring(n: int) -> Ring:
    return Ring(n)


def kernels(ring: Ring, coupling: str, lam: float):
    """Die Paarkerne der vier Kopplungen als Liste (Wahrscheinlichkeit, Abbildung für die Kette x, Abbildung für die Kette y).

    S: gemeinsamer Platz und gemeinsame Münze. C: gemeinsamer Platz, unabhängige Münzen. M: Metropolis mit gemeinsamem Platz und gemeinsamem u.
    (I, vollständig unabhängig, ist das Produkt der beiden Übergangsmatrizen und steht in `kpl_pairs`.)"""
    n = ring.n
    p = lam / (1 + lam)
    out = []
    if coupling == "S":
        for v in range(n):
            for heads, pr in ((True, p), (False, 1 - p)):
                f = ring.phi_heatbath(v, heads)
                out.append((pr / n, f, f))
    elif coupling == "C":
        for v in range(n):
            fh, ft = ring.phi_heatbath(v, True), ring.phi_heatbath(v, False)
            for fx, px in ((fh, p), (ft, 1 - p)):
                for fy, py in ((fh, p), (ft, 1 - p)):
                    out.append((px * py / n, fx, fy))
    elif coupling == "M":
        pa, pr = ring.rates("metropolis", lam)
        cuts = sorted({0.0, pa, pr, 1.0})
        segments = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]
        for v in range(n):
            for lo, hi in segments:
                f = ring.phi_metropolis(v, 0.5 * (lo + hi), lam)
                out.append(((hi - lo) / n, f, f))
    else:
        raise ValueError(coupling)
    return out


# --- Einzelschritte auf Bitmasken (für die Läufe) ----------------------------------------------------------------------------------------------

def heatbath_step(x: int, v: int, heads: bool, nbm: int) -> int:
    """Heat-bath-Aktualisierung am Platz v: Kopf belegt, wenn beide Nachbarn leer sind, sonst wird der Platz leer gesetzt."""
    bit = 1 << v
    if heads and (x & nbm) == 0:
        return x | bit
    return x & ~bit


def metropolis_step(x: int, v: int, u: float, nbm: int, pa: float, pr: float) -> int:
    """Metropolis-Aktualisierung am Platz v mit Zufallszahl u: belegen, wenn frei, beide Nachbarn leer und u < pa; leeren, wenn belegt und u < pr."""
    bit = 1 << v
    if x & bit:
        return x & ~bit if u < pr else x
    if (x & nbm) == 0 and u < pa:
        return x | bit
    return x


def lucas(n: int) -> int:
    """Zahl der Zustände des n-Rings: die Lucas-Zahl L_n (L_1 = 1, L_2 = 3, L_n = L_{n−1} + L_{n−2}); n = 4: 7, n = 8: 47, n = 40: 228 826 127."""
    a, b = 1, 3
    for _ in range(n - 1):
        a, b = b, a + b
    return a
