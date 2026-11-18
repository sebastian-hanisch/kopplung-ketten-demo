"""Vorgerechnete Dateien: Aufbau, Eigenschaften und Neurechnung (n ≤ 8 vollständig, n = 10 an Stichproben; vollständig mit `python tools/kpl_make_curves.py --check --lang`).

Die Daten (`kpl_curves.json`, `kpl_scaling.json`) sind eingefroren und in App und README als „vorgerechnet“ gekennzeichnet; dieser Test stellt sicher, dass sie die Rechnung der Demo wiedergeben."""

import math

import numpy as np
import pytest

import kpl_chain as K
import kpl_exact as E
import kpl_make_curves as MC
import kpl_make_scaling as MS
import kpl_pairs as P
import kpl_reference as REF
import kpl_sim as S


def _bad(a, b, path="x"):
    bad = []
    MC._same(a, b, path, bad)
    return bad


def test_zellen_n_bis_6_gleich_neurechnung():
    for n in (4, 6):
        for lam in MC.LAMS:
            assert _bad(MC.compute_cell(n, lam), REF.cell(n, lam)) == [], (n, lam)


@pytest.mark.parametrize("lam", [1.0, 5.0, 20.0])
def test_zellen_n8_gleich_neurechnung(lam):
    assert _bad(MC.compute_cell(8, lam), REF.cell(8, lam)) == []


def test_kopplungen_groesse_ungerade_gleich_neurechnung():
    cur = REF.curves()
    for (n, lam) in ((6, 1.0), (6, 5.0), (8, 1.0)):
        assert _bad(MC.compute_kopplungen(n, lam), cur["kopplungen"][f"{n}|{lam:g}"]) == [], (n, lam)
    for n in (4, 6, 8):
        assert _bad(MC.compute_groesse(n), cur["groesse"][str(n)]) == []
    for (n, lam) in ((7, 1.0), (9, 5.0)):
        assert _bad(MC.compute_odd(n, lam), cur["ungerade"][f"{n}|{lam:g}"]) == []


def test_zellen_n10_stichproben_gegen_eigene_rechnung():
    """n = 10 ist teuer (15 006 Paare, bis 35 000 Schritte): die Mischzeiten aus der Spektralzerlegung und die ersten 60 Werte der Kurve werden neu gerechnet, der Rest mit `--check --lang`."""
    ring = K.get_ring(10)
    for lam in (1.0, 5.0):
        c = REF.cell(10, lam)
        spec = E.Spectrum(ring.matrix("heatbath", lam), ring.pi(lam))
        assert (c["t_mix"], c["t_bar"], c["T_mix_e"]) == E.mixing_times(spec)
        pc = P.PairChain(ring, "S", lam)
        st = pc.pair(ring.index(ring.top), ring.index(ring.bot))
        ex, _ = pc.survival(st, t_stop=0.0, t_cap=60)
        for t, s in zip(c["grid"], c["surv"]):
            if t <= 60:
                assert math.isclose(ex[t], s, rel_tol=1e-9, abs_tol=1e-12), (lam, t)
        assert math.isclose(c["E_tau"], pc.mean_times(st), rel_tol=1e-9)


def test_aufbau_und_eigenschaften_der_zellen():
    cur = REF.curves()
    assert len(cur["zellen"]) == 24 and len(cur["kopplungen"]) == 4 and len(cur["ungerade"]) == 4 and set(cur["groesse"]) == {"4", "6", "8", "10"}
    for key, c in cur["zellen"].items():
        g = c["grid"]
        assert g == sorted(set(g)) and len(c["surv"]) == len(c["d"]) == len(c["dbar"]) == len(g)
        assert all(a >= b - 1e-12 for a, b in zip(c["surv"], c["surv"][1:]))          # P(τ > t) fällt
        assert all(a >= b - 1e-9 for a, b in zip(c["d"], c["d"][1:])) and all(a >= b - 1e-9 for a, b in zip(c["dbar"], c["dbar"][1:]))
        q = c["quant"]
        assert q["0.25"] <= q["0.5"] <= q["0.75"] <= q["0.9"] <= q["0.99"]
        assert c["t_c_eps"]["0.25"] == q["0.75"]                                      # t_c(1/4) ist das 75-%-Quantil
        assert c["N"] == K.lucas(c["n"]) and c["ell"] == c["n"]
    c = REF.cell(10, 20.0)
    assert len(c["surv_full"]) > 20000 and c["surv_full"][0] == 1.0 and c["surv_full"][-1] < 1e-4
    assert c["surv_full"][c["t_c_eps"]["0.25"]] <= 0.25 < c["surv_full"][c["t_c_eps"]["0.25"] - 1]


def test_survival_at_interpoliert_zwischen_stuetzstellen_und_ist_exakt_mit_vollkurve():
    c = REF.cell(8, 5.0)
    g = c["grid"]
    t = g[5]
    assert REF.survival_at(c, t) == c["surv"][5]
    i = len(g) - 3
    mid = (g[i] + g[i + 1]) // 2
    assert min(c["surv"][i], c["surv"][i + 1]) <= REF.survival_at(c, mid) <= max(c["surv"][i], c["surv"][i + 1])
    cf = REF.cell(10, 20.0)
    assert REF.survival_at(cf, 1234) == cf["surv_full"][1234]


def test_laeufe_tabelle_aufbau_und_neurechnung_zweier_zeilen():
    rows = REF.scaling_rows()
    assert len(rows) == 19 and all(r["censored"] == 0 for r in rows)
    assert REF.scaling()["meta"]["start_top_gleich_alle_starts_n_bis_12"] is True
    assert REF.scaling_row(20.0, 32) is None
    for r in rows:
        assert r["N"] == K.lucas(r["n"]) and r["q75"] <= r["q90"] and r["se"] > 0
        assert (r["t_mix"] is None) == (r["n"] > 24)
        assert r["q75_ci"][0] <= r["q75"] <= r["q75_ci"][1]
    for (lam, n) in ((1.0, 8), (5.0, 8)):
        row = REF.scaling_row(lam, n)
        top, bot = K.peak_masks(n)
        taus, cens, _ = S.sample_taus(n, lam, "S", 200, row["seed"], top, bot, MS.CAP)
        a = np.array(taus, dtype=float)
        z = (a.mean() - row["E_tau"]) / math.sqrt(a.var(ddof=1) / 200 + row["se"] ** 2)
        assert abs(z) < 4 and cens == 0
        assert MS.t_mix_full(n, lam) == row["t_mix"]
        # dieselben ersten 200 Läufe wie in der Datei: der Strom ist bitgleich (Zufalls-Seed der Zeile)
        again, _, _ = S.sample_taus(n, lam, "S", 200, row["seed"], top, bot, MS.CAP)
        assert again == taus


def test_mischzeit_aus_dem_start_der_geraden_plaetze_gleich_voller_rechnung_bis_n_12():
    for lam, n in ((1.0, 8), (5.0, 8), (1.0, 12)):
        assert MS.t_mix_top(n, lam) == MS.t_mix_full(n, lam)
