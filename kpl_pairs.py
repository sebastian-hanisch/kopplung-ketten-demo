"""Die Kopplung als Markov-Kette auf Paaren von Zuständen: exakte Verteilung und Erwartung der Treffzeit τ.

Zwei Kopien (X_t), (Y_t) derselben Kette laufen mit gemeinsamer Zufallsquelle, bis sie sich treffen (τ = min{t : X_t = Y_t}); danach laufen sie gleich weiter. Solange X_t ≠ Y_t
ist das Paar (X_t, Y_t) selbst eine Markov-Kette auf den N·(N−1) geordneten Paaren x ≠ y; Q sei ihre substochastische Matrix (die Diagonale ist absorbierend und fehlt).
Dann ist h_t = Q^t · 1 der Vektor der Wahrscheinlichkeiten P(τ > t) je Startpaar, und die mittleren Treffzeiten lösen (I − Q) m = 1.

Eigene Neuschreibung nach dem Muster der Paarkette in den Nachbar-Demos der Reihe (kein Import). Zähleinheit: Schritte (eine Platzaktualisierung; bei zwei Kopien zählt der
gemeinsame Schritt einmal)."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from kpl_chain import Ring, kernels


class PairChain:
    """Q auf den Paaren x ≠ y bis zum Zusammentreffen; `pair(x, y)` gibt den Index eines Paares in Q."""

    def __init__(self, ring: Ring, coupling: str, lam: float):
        self.ring = ring
        self.coupling = coupling
        self.lam = lam
        N = ring.N
        self.N = N
        ar = np.arange(N)
        X, Y = np.meshgrid(ar, ar, indexing="ij")
        X, Y = X.ravel(), Y.ravel()
        offd = X != Y
        self.O = np.flatnonzero(offd)
        comp = np.full(N * N, -1, dtype=np.int64)
        comp[self.O] = np.arange(len(self.O))
        self.comp = comp
        m = len(self.O)
        if coupling == "I":
            kind = "heatbath"
            Pm = ring.matrix(kind, lam)
            self.Q = sp.kron(Pm, Pm, format="csr")[self.O][:, self.O].tocsr()
            return
        rows, cols, vals = [], [], []
        for pr, fx, fy in kernels(ring, coupling, lam):
            nx, ny = fx[X[offd]], fy[Y[offd]]
            keep = nx != ny
            rows.append(comp[(X[offd] * N + Y[offd])[keep]])
            cols.append(comp[(nx * N + ny)[keep]])
            vals.append(np.full(int(keep.sum()), pr))
        self.Q = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m, m))

    @property
    def n_pairs(self) -> int:
        return int(self.Q.shape[0])

    def pair(self, x: int, y: int) -> int:
        """Index des Paares (x, y) (Zustandsindizes, x ≠ y) in Q."""
        i = int(self.comp[x * self.N + y])
        if i < 0:
            raise ValueError("Die beiden Zustände sind gleich: τ = 0")
        return i

    def reachable(self, start: int):
        """Boolesche Maske der von `start` aus erreichbaren Paare (mit Wahrscheinlichkeit > 0, ohne Treffen)."""
        seen = np.zeros(self.Q.shape[0], dtype=bool)
        seen[start] = True
        front = np.array([start])
        while len(front):
            nxt = np.unique(self.Q[front].indices)
            nxt = nxt[~seen[nxt]]
            seen[nxt] = True
            front = nxt
        return seen

    def curve(self, T: int, ext: int | None = None, stop: float = 1e-13):
        """(Maximum über alle Startpaare, Wert am Startpaar `ext`) für P(τ > t), t = 0, 1, 2, …; kürzer als T+1, sobald das Maximum unter `stop` fällt."""
        h = np.ones(self.Q.shape[0])
        mx, ex = [1.0], [1.0]
        for _ in range(T):
            h = self.Q @ h
            m = float(h.max())
            mx.append(m)
            ex.append(float(h[ext]) if ext is not None else m)
            if m < stop:
                break
        return np.array(mx), np.array(ex)

    def survival(self, start: int, t_stop: float = 1e-4, t_cap: int = 200_000):
        """P(τ > t) vom Startpaar `start` aus für t = 0, 1, … bis die Kurve unter `t_stop` liegt (oder t_cap erreicht ist); dazu das Maximum über alle Paare."""
        h = np.ones(self.Q.shape[0])
        ex, mx = [1.0], [1.0]
        for _ in range(t_cap):
            h = self.Q @ h
            ex.append(float(h[start]))
            mx.append(float(h.max()))
            if ex[-1] < t_stop:
                break
        return np.array(ex), np.array(mx)

    def mean_times(self, start: int | None = None):
        """Mittlere Treffzeit E[τ] je Startpaar aus (I − Q) m = 1; `None`, wenn von dort aus nicht jedes erreichbare Paar mit Sicherheit trifft (E[τ] = ∞).

        Mit `start` wird nur auf der von dort erreichbaren Paarmenge gelöst (und der Wert für `start` zurückgegeben)."""
        Q = self.Q
        if start is not None:
            if self.never_meets(start):
                return None
            keep = np.flatnonzero(self.reachable(start))
            Q = Q[keep][:, keep].tocsr()
            pos = int(np.searchsorted(keep, start))
        m = Q.shape[0]
        A = (sp.identity(m, format="csr") - Q).tocsr()
        b = np.ones(m)
        x, info = spla.bicgstab(A, b, rtol=1e-13, atol=0, maxiter=50000)
        if info != 0 or not np.all(np.isfinite(x)):
            if m > 4000:
                return None
            with np.errstate(all="ignore"):
                x = spla.spsolve(A.tocsc(), b)
        x = np.asarray(x, dtype=float)
        ok = np.all(np.isfinite(x)) and x.min() > 0 and x.max() < 1e12 and np.abs(A @ x - b).max() < 1e-8 * x.max()
        if not ok:
            return None
        return float(x[pos]) if start is not None else x

    def never_meets(self, start: int) -> bool:
        """Wahr, wenn von `start` aus kein erreichbares Paar auf die Diagonale übergehen kann (alle Zeilensummen von Q gleich 1)."""
        keep = self.reachable(start)
        rs = np.asarray(self.Q.sum(axis=1)).ravel()[keep]
        return bool(rs.min() > 1 - 1e-12)


def tail_quantile(curve, eps: float):
    """Kleinstes t mit curve[t] <= eps (Schranke t_c(ε) aus der Kopplung); `None`, wenn die Kurve vorher endet."""
    idx = np.flatnonzero(np.asarray(curve) <= eps)
    return int(idx[0]) if len(idx) else None
