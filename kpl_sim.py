"""Gekoppelte Läufe: zwei Kopien des Rings mit gemeinsamer Zufallsquelle, bis sie sich treffen (nur ganzzahliger SplitMix64, kein Python-`hash()`, kein numpy-Zufall).

Vier Kopplungen derselben Kette (nach dem Treffen laufen beide Kopien gleich weiter, hier wird beim Treffen abgebrochen):
* **S** gemeinsamer Platz und gemeinsame Münze (Heat-bath),
* **C** gemeinsamer Platz, unabhängige Münzen (Heat-bath),
* **I** beide Kopien vollständig unabhängig bis zum Treffen (Heat-bath),
* **M** Metropolis mit gemeinsamem Platz und gemeinsamem u.
Jede der vier ist eine Kopplung: für sich betrachtet läuft jede Kopie genau die Kette (Randketten gleich der Übergangsmatrix, geprüft in tests/test_oracle_kopplung_ketten.py).
Zähleinheit: Schritte (eine Platzaktualisierung, bei zwei Kopien zählt der gemeinsame Schritt einmal) und Läufe (jeder Lauf zieht ein τ)."""

from dataclasses import dataclass

from kpl_chain import is_independent, metropolis_step, neighbour_mask
from kpl_rng import SplitMix64

COUPLINGS = ("S", "C", "I", "M")


def make_step(rng: SplitMix64, n: int, lam: float, coupling: str):
    """Liefert `step(x, y) -> (x', y')`: ein gekoppelter Schritt aus Zufallszahlen von `rng`."""
    nbm = [neighbour_mask(n, v) for v in range(n)]
    bits = [1 << v for v in range(n)]
    below, rnd = rng.below, rng.random
    p = lam / (1 + lam)

    if coupling == "S":
        def step(x, y):
            v = below(n)
            b, nb = bits[v], nbm[v]
            if rnd() < p:
                return (x | b if not x & nb else x & ~b), (y | b if not y & nb else y & ~b)
            return x & ~b, y & ~b
    elif coupling == "C":
        def step(x, y):
            v = below(n)
            b, nb = bits[v], nbm[v]
            hx = rnd() < p
            hy = rnd() < p
            x = (x | b if not x & nb else x & ~b) if hx else x & ~b
            y = (y | b if not y & nb else y & ~b) if hy else y & ~b
            return x, y
    elif coupling == "I":
        def step(x, y):
            v = below(n)
            b, nb = bits[v], nbm[v]
            if rnd() < p:
                x = x | b if not x & nb else x & ~b
            else:
                x &= ~b
            v = below(n)
            b, nb = bits[v], nbm[v]
            if rnd() < p:
                y = y | b if not y & nb else y & ~b
            else:
                y &= ~b
            return x, y
    elif coupling == "M":
        pa, pr = min(1.0, lam), min(1.0, 1.0 / lam)

        def step(x, y):
            v = below(n)
            u = rnd()
            nb = nbm[v]
            return metropolis_step(x, v, u, nb, pa, pr), metropolis_step(y, v, u, nb, pa, pr)
    else:
        raise ValueError(coupling)
    return step


@dataclass(frozen=True)
class Run:
    tau: int | None          # Treffzeit; None, wenn die Obergrenze erreicht wurde, ohne dass sich die Kopien trafen
    steps: int               # tatsächlich gerechnete Schritte
    xs: tuple | None = None  # Zustände der Kette X nach jedem Schritt (Index 0: Start), nur mit record=True
    ys: tuple | None = None


def run_pair(rng: SplitMix64, n: int, lam: float, coupling: str, x0: int, y0: int, max_steps: int, record: bool = False) -> Run:
    """Ein gekoppelter Lauf vom Paar (x0, y0) aus bis zum ersten Treffen (τ = 0, wenn x0 = y0) oder bis `max_steps`."""
    step = make_step(rng, n, lam, coupling)
    x, y = x0, y0
    xs, ys = ([x], [y]) if record else (None, None)
    t = 0
    while x != y and t < max_steps:
        x, y = step(x, y)
        t += 1
        if record:
            xs.append(x)
            ys.append(y)
    return Run(t if x == y else None, t, tuple(xs) if record else None, tuple(ys) if record else None)


def run_budget(R: int, e_tau: float, budget: int) -> int:
    """Wie viele von R Läufen das Schrittbudget erlaubt, wenn ein Lauf im Mittel e_tau Schritte braucht (mindestens 1). Wird vor dem Rechnen festgelegt, nicht aus den Läufen,
    damit die gerechneten Läufe unverzerrt bleiben."""
    return max(1, min(R, int(budget // max(1.0, e_tau))))


def sample_taus(n: int, lam: float, coupling: str, runs: int, seed: int, x0: int, y0: int, cap: int):
    """`runs` Läufe nacheinander aus einem Strom (Zufalls-Seed `seed`); die ersten k Läufe sind für jedes `runs ≥ k` dieselben. Rückgabe: (Liste der τ, Zahl der Läufe ohne Treffen
    bis `cap`, insgesamt gerechnete Schritte)."""
    rng = SplitMix64(seed)
    taus, censored, steps = [], 0, 0
    for _ in range(runs):
        r = run_pair(rng, n, lam, coupling, x0, y0, cap)
        steps += r.steps
        if r.tau is None:
            censored += 1
        else:
            taus.append(r.tau)
    return taus, censored, steps


def random_state(rng: SplitMix64, n: int) -> int:
    """Zufällige zulässige Belegung (nicht aus dem Gleichgewicht gezogen): die Plätze in zufälliger Reihenfolge durchgehen, jeden freien mit Wahrscheinlichkeit 1/2 belegen."""
    order = list(range(n))
    rng.shuffle(order)
    x = 0
    for v in order:
        if (x & neighbour_mask(n, v)) == 0 and rng.random() < 0.5:
            x |= 1 << v
    return x


def random_pair(seed: int, n: int) -> tuple[int, int]:
    """Zwei verschiedene zufällige Belegungen (Zufalls-Seed `seed`)."""
    rng = SplitMix64(seed)
    for _ in range(1000):
        x, y = random_state(rng, n), random_state(rng, n)
        if x != y:
            return x, y
    return (1, 0)


def parse_state(text: str, n: int) -> tuple[int | None, str]:
    """Belegung aus einer 0/1-Folge der Länge n lesen (Zeichen i = Platz i). Rückgabe (Maske, Fehlermeldung); bei Fehler ist die Maske None."""
    s = text.strip().replace(" ", "")
    if len(s) != n or any(c not in "01" for c in s):
        return None, f"Eine Belegung besteht aus genau {n} Zeichen 0 oder 1 (Platz 0 zuerst), zum Beispiel {'01' * (n // 2) + '0' * (n % 2)}."
    mask = sum(1 << i for i, c in enumerate(s) if c == "1")
    if not is_independent(mask, n):
        return None, "Zwei benachbarte Plätze sind belegt (der Ring ist geschlossen: Platz n−1 grenzt an Platz 0)."
    return mask, ""


def mask_text(mask: int, n: int) -> str:
    return "".join("1" if mask >> i & 1 else "0" for i in range(n))
