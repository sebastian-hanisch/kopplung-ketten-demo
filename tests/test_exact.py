"""Einheiten der exakten Rechnung: Suche der Schwelle, Quantile, Stichprobenquantil, Band, Ordnung, Stützstellen."""

import math

import numpy as np
import pytest

import kpl_chain as K
import kpl_exact as E
import kpl_pairs as P
from kpl_rng import SplitMix64


def test_first_t_findet_die_erste_schwelle():
    assert E.first_t(lambda t: 1.0 / t, 0.1) == 10
    assert E.first_t(lambda t: 0.5 ** t, 0.25) == 2
    assert E.first_t(lambda t: 1.0, 0.5, t_cap=100) is None
    assert E.first_t(lambda t: 0.0, 0.5) == 1


def test_gitterpunkte_der_kurven():
    g = E.grid_points(100)
    assert g[:40] == list(range(1, 41)) and g == sorted(set(g)) and g[-1] == 100
    assert E.grid_points(10) == list(range(1, 11))


def test_quantile_der_treffzeit():
    surv = np.array([1.0, 1.0, 0.6, 0.3, 0.2, 0.0])
    assert E.quantiles(surv, (0.25, 0.5, 0.75, 0.9, 1.0)) == {0.25: 2, 0.5: 3, 0.75: 4, 0.9: 5, 1.0: 5}
    assert P.tail_quantile(surv, 0.25) == 4 and P.tail_quantile(surv, 0.0) == 5 and P.tail_quantile(np.ones(5), 0.25) is None


def test_stichprobenquantil_und_rang():
    assert E.sample_quantile([5, 1, 9, 3], 0.75) == 5             # ⌈0,75 · 4⌉ = 3: die dritte kleinste Zahl
    assert E.sample_quantile([7], 0.75) == 7
    assert E.sample_quantile(list(range(1, 11)), 0.75) == 8       # ⌈7,5⌉ = 8


def test_inversionsverfahren_gibt_die_kleinste_stelle_mit_cdf_ueber_u():
    surv = np.array([1.0, 0.75, 0.25, 0.0])                        # F = 0, 0,25, 0,75, 1
    assert [E.draw_tau_from_survival(surv, u) for u in (0.0, 0.2, 0.25, 0.26, 0.75, 0.9)] == [0, 1, 1, 2, 2, 3]


def test_band_und_wahrscheinlichkeit_unter_einer_marke():
    surv = np.array([1.0, 0.9, 0.6, 0.3, 0.1, 0.0])
    lo, hi = E.quantile_estimate_band(surv, 1)                    # R = 1: das Quantil ist die einzige Zahl selbst
    assert (lo, hi) == (E.quantiles(surv, (0.05,))[0.05], E.quantiles(surv, (0.95,))[0.95])
    assert E.prob_estimate_below(surv, 1, 3) == pytest.approx(1 - surv[2])   # P(τ < 3) = P(τ ≤ 2)
    assert E.prob_estimate_below(surv, 5, 99) == 1.0


def test_band_wird_mit_mehr_laeufen_schmaler_und_enthaelt_das_exakte_quantil():
    surv = np.exp(-np.arange(0, 400) / 40.0)
    t_c = P.tail_quantile(surv, 0.25)
    widths = []
    for R in (10, 20, 50, 200, 1000):
        lo, hi = E.quantile_estimate_band(surv, R)
        assert lo <= t_c <= hi
        widths.append(hi - lo)
    assert widths == sorted(widths, reverse=True)


def test_verteilungsfreies_band_des_stichprobenquantils_ueberdeckt_das_wahre_quantil():
    rng = SplitMix64(9)
    true_q = -math.log(0.25) * 40.0                                # 75-%-Quantil der Exponentialverteilung mit Mittel 40
    hits = 0
    for _ in range(300):
        xs = [rng.expovariate(40.0) for _ in range(50)]
        lo, hi = E.sample_quantile_ci(xs, 0.75)
        hits += (lo is None or lo <= true_q) and (hi is None or true_q <= hi)
    assert hits >= 0.92 * 300
    assert E.sample_quantile_ci([1.0, 2.0, 3.0], 0.75)[1] is None  # bei R = 3 gibt es kein Band nach oben


def test_laengste_kette_der_ordnung_ist_n_bei_geradem_n():
    for n in (4, 6, 8, 10):
        assert E.longest_chain(K.get_ring(n)) == n
    with pytest.raises(ValueError):
        E.longest_chain(K.get_ring(5))


def test_sammler_und_hyperwuerfel_zeilen():
    row = E.hypercube_row(8)
    assert (row["t_mix"], row["t_bar"], row["t_c"]) == (12, 17, 26) and math.isclose(row["E_tau"], 8 * sum(1 / i for i in range(1, 9)))
    s = E.collector_survival(5, 200)
    assert s[0] == 1 and np.all(np.diff(s) <= 1e-15) and s[-1] < 1e-12


def test_drei_zustaende_hat_zehn_zeilen_und_den_mittelwert_3():
    ts = E.three_state_example(10)
    assert len(ts["rows"]) == 10 and ts["ell"] == 2 and ts["E_tau"] == 3


def test_vektorisiertes_inversionsverfahren_gleich_dem_skalaren():
    surv = np.exp(-np.arange(0, 300) / 30.0)
    rng = SplitMix64(4)
    us = [rng.random() for _ in range(500)]
    assert list(E.draw_taus_from_survival(surv, us)) == [E.draw_tau_from_survival(surv, u) for u in us]
