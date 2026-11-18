"""Randfälle (DEMO-PLAYBOOK §12): n = 1, 2, 3, ungerades n, gleiche Startzustände, falsche Eingaben, Budget, Ränder der Regler, Metropolis bei λ = 1."""

import pytest

import kpl_chain as K
import kpl_constants as C
import kpl_evaluation as ev
import kpl_exact as E
import kpl_pairs as P
import kpl_reference as REF
import kpl_sim as S


@pytest.mark.parametrize("n", [1, 2, 3])
def test_ring_unter_vier_plaetzen_hat_kein_gipfelpaar(n):
    with pytest.raises(ValueError):
        K.peak_masks(n)
    if n < 3:
        with pytest.raises(ValueError):
            K.ring_states(n)
    r = ev.analyse_lauf(ev.E1Settings(n=n))
    assert r.status == "fehler" and "mindestens 4 Plätze" in r.message


def test_ring_mit_drei_plaetzen_hat_vier_zustaende_aber_keine_ordnung():
    assert K.get_ring(3).N == 4 == K.lucas(3)
    with pytest.raises(ValueError):
        E.longest_chain(K.get_ring(3))


def test_ungerades_n_gipfelpaar_sind_die_maximalen_belegungen_auf_den_plaetzen_0_bis_n_minus_2():
    top, bot = K.peak_masks(7)
    assert (top, bot) == (0b0010101, 0b0101010) and K.is_independent(top, 7) and K.is_independent(bot, 7)
    r = ev.analyse_lauf(ev.E1Settings(n=7, lam=1.0))
    assert r.status == "ok" and r.is_peak


def test_gleicher_start_gibt_tau_null_ohne_lauf():
    r = ev.analyse_lauf(ev.E1Settings(start="eigene Eingabe", x_text="00000000", y_text="00000000"))
    assert r.status == "gleich" and r.run is None and "τ = 0" in r.message
    with pytest.raises(ValueError):
        P.PairChain(K.get_ring(4), "S", 1.0).pair(2, 2)


@pytest.mark.parametrize("x,y,teil", [("1010", "01010101", "Kette X"), ("10101010", "11000000", "Kette Y"), ("10101010", "0101010a", "Kette Y")])
def test_falsche_eingaben_geben_eine_meldung_statt_eines_laufs(x, y, teil):
    r = ev.analyse_lauf(ev.E1Settings(start="eigene Eingabe", x_text=x, y_text=y))
    assert r.status == "fehler" and r.message.startswith(teil) and r.run is None


def test_eigene_eingabe_gibt_einen_lauf_mit_exakter_verteilung():
    r = ev.analyse_lauf(ev.E1Settings(n=6, lam=2.0, start="eigene Eingabe", x_text="101010", y_text="010000", coupling="S"))
    assert r.status == "ok" and not r.is_peak and r.exact is not None and r.exact["source"] == "live"


def test_unabhaengige_kopplung_ueber_der_grenze_gibt_eine_meldung():
    r = ev.analyse_lauf(ev.E1Settings(n=C.UNABHAENGIG_MAX_N + 2, coupling="I"))
    assert r.status == "zu_gross" and r.run is None and "nur bis n = 8" in r.message
    assert ev.analyse_lauf(ev.E1Settings(n=C.UNABHAENGIG_MAX_N, coupling="I", lam=1.0)).status == "ok"


@pytest.mark.parametrize("n", [4, 6, 8])
def test_metropolis_bei_lambda_1_trifft_sich_nie_exakt(n):
    r = ev.analyse_lauf(ev.E1Settings(n=n, lam=1.0, coupling="M"))
    assert r.status == "nie" and r.exact["never"] and r.run.tau is None and r.run.steps == C.LAUF_MAX_SCHRITTE
    ring = K.get_ring(n)
    st = (ring.index(ring.top), ring.index(ring.bot))
    pc = P.PairChain(ring, "M", 1.0)
    assert pc.never_meets(pc.pair(*st))
    assert pc.mean_times(pc.pair(*st)) is None
    for lam in (2.0, 5.0):                                              # bei λ ≠ 1 gibt es ein u-Intervall, in dem sie sich treffen
        pl = P.PairChain(ring, "M", lam)
        assert not pl.never_meets(pl.pair(*st))


def test_metropolis_bei_lambda_1_n_ueber_der_exakten_grenze_endet_an_der_obergrenze():
    r = ev.analyse_lauf(ev.E1Settings(n=12, lam=1.0, coupling="M"))
    assert r.status == "ohne_treffen" and r.exact is None and r.run.tau is None


def test_raender_der_regler_laufen_durch():
    for n in (C.E1_N_MIN, C.E1_N_MAX):
        for lam in (min(C.LAM_GRID), max(C.LAM_GRID)):
            r = ev.analyse_lauf(ev.E1Settings(n=n, lam=lam, coupling="S"))
            assert r.status == "ok" and r.run.tau >= n


def test_ein_lauf_und_laeufe_ueber_dem_budget():
    top, bot = K.peak_masks(8)
    taus, cens, steps = S.sample_taus(8, 5.0, "S", 1, 3, top, bot, 10**6)
    assert len(taus) == 1 and cens == 0 and steps == taus[0]
    r = ev.analyse_schranke(8, 20.0, 0.25, 1000, 7)                     # E[τ] = 2224: das Budget lässt 134 Läufe zu
    assert r.runs_asked == 1000 and r.runs_done == 134 and len(r.taus) + r.censored == 134
    r = ev.analyse_schranke(4, 1.0, 0.25, 10, 7)
    assert r.runs_done == 10 and min(r.taus) >= 4


def test_matrixfrei_ohne_messzeile_und_mit_einem_lauf():
    assert ev.analyse_matrixfrei(40, 20.0, 100, 1).row is None
    r = ev.analyse_matrixfrei(8, 1.0, 10, 1)
    assert r.row is not None and r.runs_done == 10 and r.se is not None
    one = ev.analyse_matrixfrei(8, 1.0, 1, 1)
    assert one.runs_done == 1 and one.se is None and one.q75_ci is None


def test_laeufe_panel_r_1000_und_r_10():
    for R in (10, 1000):
        r = ev.analyse_laeufe(R, 50, 5)
        assert r.band[0] <= r.t_c <= r.band[1] and len(r.estimates) == 50 and 0 <= r.n_below_mix <= 50


def test_kopplungen_ohne_auswahl_und_nur_gipfelpaar():
    assert ev.kopplung_rows(8, 1.0, (), "alle Paare") == []
    rows = ev.kopplung_rows(8, 1.0, ("S", "M"), "nur Gipfelpaar")
    assert [r["coupling"] for r in rows] == ["S", "M"] and rows[1]["never"] and rows[1]["t_c"] is None
    assert rows[0]["t_c"] == REF.kopplungen(8, 1.0)["S"]["t_c_gipfel"] == 66


def test_zellen_fuer_nicht_vorgerechnete_n_und_lambda_gibt_es_nicht():
    assert not REF.has_cell(12, 5.0) and not REF.has_cell(8, 3.0)
    assert ev.exact_pair_distribution(12, 5.0, "S", *K.peak_masks(12)) is None
