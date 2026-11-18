"""Zahlen im deutschen Format: Dezimalkomma, Gruppierung mit Leerzeichen ab fünf Stellen (DIN 5008)."""

from fractions import Fraction


def ganz(n) -> str:
    """16796 → „16 796“, 1430 → „1430“."""
    n = int(round(n))
    s = str(abs(n))
    if len(s) >= 5:
        gruppen = []
        while s:
            gruppen.append(s[-3:])
            s = s[:-3]
        s = " ".join(reversed(gruppen))
    return ("-" if n < 0 else "") + s


def dez(x: float, stellen: int = 2) -> str:
    return f"{x:.{stellen}f}".replace(".", ",")


def faktor(x: float) -> str:
    """Verhältnis mit zwei Stellen: 2,26-fach wird als „2,26“ geschrieben."""
    return dez(x, 2)


def prozent(anteil: float, stellen: int = 1) -> str:
    return f"{dez(anteil * 100, stellen)} %"


def bruch(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)


def lam(x: float) -> str:
    """λ ohne überflüssige Nullen: 0,5 · 1 · 20."""
    return dez(x, 1).rstrip("0").rstrip(",") if x != int(x) else str(int(x))


def eps(x: float) -> str:
    """ε als Bruch oder Dezimalzahl: 1/4 · 0,1 · 0,01 · 0,001."""
    return "1/4" if abs(x - 0.25) < 1e-12 else dez(x, 3).rstrip("0").rstrip(",")
