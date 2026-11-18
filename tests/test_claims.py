"""Jede Zahl der README steht hier (Tabelle „Zahl → Test“ in der README, Spalte „Test“).

Ganzzahlen, Brüche und exakte Größen exakt bzw. auf Gleitkomma-Toleranz; Zufallszahlen aus Läufen mit festem Zufalls-Seed als Band (|z| < 4 gegen die exakte Größe).
Gemessen wird mit denselben Funktionen wie in der App (`kpl_evaluation`) und mit der vorgerechneten Datei, die `tests/test_reference.py` gegen die Neurechnung prüft."""

import math
import re
import subprocess
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

import kpl_chain as K
import kpl_evaluation as ev
import kpl_exact as E
import kpl_pairs as P
import kpl_reference as REF
import kpl_sim as S

ROOT = Path(__file__).resolve().parent.parent


def cell(n, lam):
    return REF.cell(n, lam)


def tc(c, eps="0.25"):
    return c["t_c_eps"][eps]


def r2(x):
    return round(x + 1e-12, 2)


# ---------------------------------------------------------------------------------------------------------------------------------------

def test_k1_kleinstfall_n4_lambda1():
    c = cell(4, 1.0)
    assert K.lucas(4) == 7
    assert (c["t_mix"], c["t_bar"], c["T_mix_e"], tc(c)) == (12, 21, 16, 25)
    assert math.isclose(c["E_tau"], 135 / 7, rel_tol=1e-11) and round(c["E_tau"], 2) == 19.29
    assert (r2(tc(c) / c["t_mix"]), r2(tc(c) / c["t_bar"])) == (2.08, 1.19)
    assert r2(4 * c["E_tau"] / c["t_mix"]) == 6.43 and round(4 * 135 / 7, 1) == 77.1
    ring = K.get_ring(4)
    pc = P.PairChain(ring, "S", 1.0)
    st = pc.pair(ring.index(ring.top), ring.index(ring.bot))
    assert int(pc.reachable(st).sum()) == 19 and pc.n_pairs == 42
    ex, _ = pc.survival(st, t_stop=0.0, t_cap=6)
    for t, bruch in ((4, F(63, 64)), (5, F(977, 1024)), (6, F(1871, 2048))):
        assert math.isclose(ex[t], float(bruch), abs_tol=1e-13)
    assert ex[3] > 1 - 1e-12                                         # τ ≥ 4


def test_k2_preis_ueber_alle_24_zellen():
    p = ev.price_stats()
    assert p["cells"] == 24
    assert [r2(x) for x in p["c_over_mix"]] == [1.89, 2.28, 2.12]
    assert [r2(x) for x in p["bar_over_mix"]] == [1.43, 1.99, 1.76]
    assert [r2(x) for x in p["c_over_bar"]] == [1.00, 1.41, 1.21]
    assert [r2(x) for x in p["e_over_mix"][:2]] == [1.45, 1.77]
    assert [r2(x) for x in p["four_e_over_mix"][:2]] == [5.79, 7.07]


@pytest.mark.parametrize("n,lam,werte,faktor,faktor_bar", [
    (4, 20.0, (671, 1338, 966, 1342), 2.00, 1.00),
    (6, 1.0, (21, 34, 26, 44), 2.10, None),
    (8, 1.0, (31, 48, 38, 66), 2.13, None),
    (8, 5.0, (153, 280, 209, 346), 2.26, None),
    (8, 20.0, (1403, 2737, 1993, 2983), 2.13, None),
    (10, 5.0, (208, 364, 274, 474), 2.28, None),
    (10, 20.0, (1805, 3468, 2539, 3967), 2.20, None),
])
def test_k3_ausgewaehlte_zellen(n, lam, werte, faktor, faktor_bar):
    c = cell(n, lam)
    assert (c["t_mix"], c["t_bar"], c["T_mix_e"], tc(c)) == werte
    assert r2(tc(c) / c["t_mix"]) == faktor
    if faktor_bar is not None:
        assert r2(tc(c) / c["t_bar"]) == faktor_bar


def test_k4_strengeres_epsilon():
    c = cell(8, 5.0)
    assert [c["t_mix_eps"][e] for e in ("0.25", "0.1", "0.01", "0.001")] == [153, 322, 748, 1175]
    assert [c["t_bar_eps"][e] for e in ("0.25", "0.1", "0.01", "0.001")] == [280, 450, 877, 1303]
    assert [tc(c, e) for e in ("0.25", "0.1", "0.01", "0.001")] == [346, 516, 943, 1369]
    assert [r2(tc(c, e) / c["t_mix_eps"][e]) for e in ("0.25", "0.1", "0.01", "0.001")] == [2.26, 1.60, 1.26, 1.17]
    c10 = cell(10, 5.0)
    assert [r2(tc(c10, e) / c10["t_mix_eps"][e]) for e in ("0.25", "0.1", "0.01", "0.001")] == [2.28, 1.65, 1.29, 1.18]


def test_k5_verteilung_von_tau_n8_lambda5():
    c = cell(8, 5.0)
    assert [c["quant"][k] for k in ("0.25", "0.5", "0.75", "0.9", "0.99")] == [129, 214, 346, 516, 943]
    assert round(c["E_tau"], 2) == 265.32
    ring = K.get_ring(8)
    pc = P.PairChain(ring, "S", 5.0)
    st = pc.pair(ring.index(ring.top), ring.index(ring.bot))
    ex, mx = pc.survival(st, t_stop=1e-6)
    assert ex[7] > 1 - 1e-12 and ex[8] < 1 - 1e-8                    # kleinstes mögliches τ = 8
    assert np.abs(mx - ex).max() < 1e-12                             # das Gipfelpaar ist das schlechteste Startpaar


def test_k6_hyperwuerfel():
    rows = {r["m"]: r for r in ev.hypercube_table()}
    assert [rows[m]["t_mix"] for m in (8, 12, 50, 100, 400)] == [12, 20, 120, 275, 1378]
    assert [rows[m]["t_bar"] for m in (8, 12, 50, 100, 400)] == [17, 28, 154, 343, 1654]
    assert [rows[m]["t_c"] for m in (8, 12, 50, 100, 400)] == [26, 44, 257, 584, 2893]
    assert [r2(rows[m]["c_over_mix"]) for m in (8, 12, 50, 100, 400)] == [2.17, 2.20, 2.14, 2.12, 2.10]
    assert [round(rows[m]["E_tau"], 1) for m in (8, 12, 50, 100, 400)] == [21.7, 37.2, 225.0, 518.7, 2628.0]


def test_k7_welche_kopplung_n8_lambda1():
    k = REF.kopplungen(8, 1.0)
    assert (k["S"]["t_mix"], k["S"]["t_c_alle"], k["I"]["t_c_alle"], k["C"]["t_c_alle"]) == (31, 66, 208, 328)
    assert [r2(k[c]["t_c_alle"] / 31) for c in "SIC"] == [2.13, 6.71, 10.58]
    assert k["M"]["never"] and k["M"]["t_mix"] == 15 and k["M"]["t_c_alle"] is None


def test_k8_vier_zellen_s_c_i_und_metropolis():
    zellen = [(6, 1.0), (6, 5.0), (8, 1.0), (8, 5.0)]
    i_s = [r2(REF.kopplungen(n, lam)["I"]["t_c_alle"] / REF.kopplungen(n, lam)["S"]["t_c_alle"]) for n, lam in zellen]
    c_s = [r2(REF.kopplungen(n, lam)["C"]["t_c_alle"] / REF.kopplungen(n, lam)["S"]["t_c_alle"]) for n, lam in zellen]
    assert i_s == [1.95, 1.24, 3.15, 1.52] and c_s == [2.91, 1.32, 4.97, 1.65]
    for n, lam in zellen:
        k = REF.kopplungen(n, lam)
        assert k["S"]["t_c_alle"] < min(k["I"]["t_c_alle"], k["C"]["t_c_alle"])
    m6, m8 = REF.kopplungen(6, 5.0)["M"], REF.kopplungen(8, 5.0)["M"]
    assert (m6["t_c_alle"], m6["t_mix"], r2(m6["t_c_alle"] / m6["t_mix"])) == (219, 87, 2.52)
    assert (m8["t_c_alle"], m8["t_mix"], r2(m8["t_c_alle"] / m8["t_mix"])) == (332, 127, 2.61)


def test_k9_groesse_bei_lambda_1():
    rows = ev.groesse_rows()
    assert [r["n"] for r in rows] == [4, 6, 8, 10] and [r["N"] for r in rows] == [7, 18, 47, 123]
    assert [r["S"] for r in rows] == [25, 44, 66, 90] and [r["I"] for r in rows] == [33, 86, 208, 499] and [r["C"] for r in rows] == [46, 128, 328, 814]
    assert [r2(r["I_over_N"]) for r in rows] == [4.71, 4.78, 4.43, 4.06]
    assert all(r["C"] > r["I"] > r["S"] for r in rows)


def test_k10_metropolis_lambda_1_hat_keinen_uebergang_auf_die_diagonale():
    for n, paare in ((4, 42), (6, 306), (8, 2162), (10, 15006)):
        ring = K.get_ring(n)
        assert ring.N * (ring.N - 1) == paare
        pc = P.PairChain(ring, "M", 1.0)
        assert pc.n_pairs == paare
        zeilensummen = np.asarray(pc.Q.sum(axis=1)).ravel()
        assert (zeilensummen < 1 - 1e-12).sum() == 0                 # 0 von N(N−1) Paaren haben einen Übergang auf die Diagonale
    ring = K.get_ring(8)
    for lam in (2.0, 5.0):
        pc = P.PairChain(ring, "M", lam)
        assert (np.asarray(pc.Q.sum(axis=1)).ravel() < 1 - 1e-12).sum() > 0
    assert REF.kopplungen(8, 1.0)["M"]["t_mix"] == 15


def test_k11_zwei_startzustaende():
    m = ev.monotony_stats()
    assert (m["even"][0], m["odd"][0], m["metropolis"][0], m["total"]) == (24, 4, 2, 30)
    assert m["max"] < 2.2e-15 and m["max"] > 1e-15                 # „2e-15“
    odd = {k: v for k, v in REF.curves()["ungerade"].items()}
    assert [r2(odd[k]["t_c_alle"] / odd[k]["t_mix"]) for k in ("7|1", "7|5", "9|1", "9|5")] == [2.27, 2.51, 2.21, 2.61]
    assert all(v["t_c_alle"] == v["t_c_gipfel"] for v in odd.values())


def test_k12_ungleichungen_und_satz_5():
    st = ev.satz5_stats()
    assert st["points"] == 1379
    assert (st["d_le_P"], st["left"], st["right"], st["dbar_2d"], st["wrong_d"]) == (0, 0, 0, 0, 0)


def test_k13_drei_zustaende():
    ts = ev.three_state()
    r1 = ts["rows"][0]
    assert (r1["tail_over_ell"], r1["d"], r1["dbar"]) == (F(1, 2), F(1, 3), F(1, 2))
    assert all(r["dbar"] == F(1, 2 ** r["k"]) for r in ts["rows"]) and ts["E_tau"] == 3


def test_k14_laeufetabelle_zeilen():
    z = {(r["lam"], r["n"]): r for r in REF.scaling_rows()}
    assert len(z) == 19 and all(r["censored"] == 0 for r in REF.scaling_rows())
    erwartet = {
        (1.0, 8): (52.9, 0.7, 66), (1.0, 24): (237.6, 2.2, 287), (1.0, 40): (451.9, 4.9, 541),
        (5.0, 8): (269.2, 4.4, 354), (5.0, 24): (1252.8, 14.3, 1564), (5.0, 40): (2538.1, 37.6, 3091),
        (20.0, 12): (3825.6, 66.3, 4986), (20.0, 24): (10085.2, 235.5, 12743),
    }
    for k, (e, se, q75) in erwartet.items():
        r = z[k]
        assert (round(r["E_tau"], 1), round(r["se"], 1), r["q75"]) == (e, se, q75), k
    assert REF.scaling_row(5.0, 40)["N"] == 228_826_127 and K.lucas(24) == 103_682 and K.lucas(16) == 2207


def test_k14b_laeufetabelle_gegen_exakte_erwartung_im_band():
    """Die vorgerechneten Zeilen mit exakter Verteilung (n = 8, λ = 1 und 5) liegen im Band gegen das exakte E[τ]."""
    for lam in (1.0, 5.0):
        r = REF.scaling_row(lam, 8)
        c = cell(8, lam)
        assert abs(r["E_tau"] - c["E_tau"]) < 4 * r["se"]
        assert r["q75_ci"][0] <= c["quant"]["0.75"] <= r["q75_ci"][1]          # das exakte 75-%-Quantil liegt im 95-%-Band der Läufe


def test_k15_q75_gegen_exakte_mischzeit():
    rows = [r for r in ev.scaling_table() if r["q75_over_mix"] is not None]
    assert len(rows) == 15 and all(r["n"] <= 24 for r in rows)
    st = ev.scaling_stats()
    assert (st["known"], r2(st["lo"]), r2(st["hi"])) == (15, 2.02, 2.40)
    z = {(r["lam"], r["n"]): r for r in rows}
    assert z[(5.0, 24)]["t_mix"] == 705 and r2(z[(5.0, 24)]["q75_over_mix"]) == 2.22


def test_k16_e_tau_gegen_n_ln_n_und_t_rel():
    rows = ev.scaling_table()
    l1 = [r for r in rows if r["lam"] == 1.0]
    l5 = [r for r in rows if r["lam"] == 5.0]
    assert (r2(min(r["E_over_nln"] for r in l1)), r2(max(r["E_over_nln"] for r in l1))) == (3.06, 3.18)
    assert (r2(min(r["E_over_nln"] for r in l5)), r2(max(r["E_over_nln"] for r in l5))) == (16.04, 17.20)
    assert (r2(l1[0]["E_over_rel"]), r2(l1[-1]["E_over_rel"])) == (1.93, 3.31)
    assert (r2(min(r["E_over_rel"] for r in l5)), r2(max(r["E_over_rel"] for r in l5))) == (1.45, 2.73)
    assert all(a["E_over_rel"] < b["E_over_rel"] for a, b in zip(l1, l1[1:]))      # wächst mit n
    # t_rel = n/g(λ): die Formel gegen den zweitgrößten Eigenwert der Heat-bath-Matrix (n = 4 bis 10, gerade)
    import kpl_make_scaling as MS
    for n in (4, 6, 8, 10):
        ring = K.get_ring(n)
        for lam in (1.0, 5.0):
            ev_ = E.Spectrum(ring.matrix("heatbath", lam), ring.pi(lam)).ev
            assert math.isclose(1.0 / (1.0 - ev_[1]), n / MS.g_closed(lam), rel_tol=1e-9), (n, lam)


def test_k17_stichproben_tv_untergrenze():
    assert round(ev.tv_lower_bound_sample(K.lucas(24), 10_000), 2) == 0.90
    assert round(ev.tv_lower_bound_sample(K.lucas(40), 10**5), 4) == 0.9996


def test_k18_wie_viele_laeufe():
    c = cell(10, 20.0)
    surv = REF.full_curve(10, 20.0)
    assert c["t_mix"] == 1805 and P.tail_quantile(surv, 0.25) == 3967 == tc(c)
    erwartet = {10: (2261, 6499), 20: (2521, 5370), 50: (3091, 5023), 200: (3485, 4447), 1000: (3750, 4183)}
    for R, band in erwartet.items():
        assert E.quantile_estimate_band(surv, R) == band, R
    p = {R: E.prob_estimate_below(surv, R, 1805) for R in (10, 20, 50, 200, 1000)}
    assert round(100 * p[10], 2) == 1.02 and round(100 * p[20], 2) == 0.12 and p[50] < 1e-4 and p[1000] < 1e-100
    # die gezogenen Wiederholungen folgen dieser Verteilung: 1000 Wiederholungen bei R = 10, Anteil unter t_mix im Binomialband um 1,02 %
    r = ev.analyse_laeufe(10, 1000, 6101)
    assert abs(r.n_below_mix - 1000 * p[10]) < 4 * math.sqrt(1000 * p[10] * (1 - p[10]))
    lo, hi = r.band
    assert np.mean([lo <= e <= hi for e in r.estimates]) > 0.85     # das exakte 5-%-bis-95-%-Band enthält etwa 90 % der Schätzungen
    # Schätzungen gleich exakter Verteilung: Mittel der gezogenen Schätzungen gegen das exakte Mittel von q̂ (Band)
    cdf = E.quantile_estimate_cdf(surv, 10)
    pmf = np.diff(np.concatenate([[0], cdf]))
    mean_q = float((np.arange(len(pmf)) * pmf).sum())
    sd_q = math.sqrt(float(((np.arange(len(pmf)) - mean_q) ** 2 * pmf).sum()))
    assert abs(np.mean(r.estimates) - mean_q) < 4 * sd_q / math.sqrt(1000)


def test_k19_simulierte_laeufe_gegen_exakt_in_der_app():
    """Die Treppe der App (Experiment 2) für n = 8, λ = 5 und Kopplung S: E[τ] der Läufe im Band um das exakte 265,32."""
    r = ev.analyse_schranke(8, 5.0, 0.25, 1000, 6101)
    a = np.array(r.taus, dtype=float)
    assert r.runs_done == 1000 and r.censored == 0
    assert abs(a.mean() - 265.32130407) < 4 * a.std(ddof=1) / math.sqrt(len(a))


# ---------------------------------------------------------------------------------------------------------------------------------------

def test_readme_nennt_die_testzahl_der_suite():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"], cwd=ROOT, capture_output=True, text=True, timeout=300).stdout
    m = re.search(r"(\d+) tests? collected", out) or re.search(r"^(\d+) tests", out, re.M)
    n = int(m.group(1)) if m else len([ln for ln in out.splitlines() if "::" in ln])
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"**{n} Tests**" in readme, f"die README nennt nicht {n} Tests"


def test_jede_testfunktion_der_readme_tabelle_existiert():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    namen = set(re.findall(r"`(test_k\d+b?)`", readme))
    quelle = Path(__file__).read_text(encoding="utf-8")
    assert len(namen) >= 17
    for n in namen:
        assert re.search(rf"def {n}_", quelle), n
