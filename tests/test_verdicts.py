"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jeder Zweig und jede Zahl im Text steht hier. Ein Verdikt ist eine reine Funktion (Art, Text)."""

import numpy as np

import kpl_constants as C
import kpl_evaluation as ev
import kpl_sim as S


def _run(tau, steps=None):
    steps = tau if steps is None else steps
    return S.Run(tau, steps, (0,) * (steps + 1), (0,) * (steps + 1))


EXACT = {"never": False, "surv": np.array([1.0, 1.0, 0.5, 0.25, 0.0]), "mean": 2.75, "quant": {0.25: 2, 0.5: 2, 0.75: 3, 0.9: 4, 0.99: 4}, "source": "live"}


def _res(status, coupling="S", lam=5.0, n=8, run=None, exact=None, peak=True, msg=""):
    return ev.E1Result(ev.E1Settings(n=n, lam=lam, coupling=coupling), status, msg, 1, 2, run, exact, peak)


def test_lauf_fehler_gleich_und_zu_gross_geben_die_meldung_weiter():
    assert ev.verdict_lauf(_res("fehler", msg="Kette X: falsch")) == ("error", "Kette X: falsch")
    assert ev.verdict_lauf(_res("gleich", msg="τ = 0"))[0] == "info"
    kind, text = ev.verdict_lauf(_res("zu_gross", n=12, msg="Die unabhängige Kopplung I ist nur bis n = 8 freigegeben"))
    assert kind == "warning" and "nur bis n = 8" in text


def test_lauf_mit_exakter_verteilung_nennt_quantile_mittel_und_stellung():
    kind, text = ev.verdict_lauf(_res("ok", run=_run(3), exact=EXACT))
    assert kind == "info"
    assert "n = 8, λ = 5, Kopplung gemeinsamer Platz und Münze (S)" in text and "**τ = 3 Schritten**" in text
    assert "Viertelquantil 2, Median 2, 75-%-Quantil 3, 99-%-Quantil 4 Schritte, Mittel 2,75" in text
    assert "Startpaar: Gipfelpaar; live gerechnet" in text
    assert "über 50,0 % der Läufe (P(τ < 3) = 0,500)" in text
    assert "Ein Lauf ist eine Zahl aus dieser Verteilung, kein Wert der Kette." in text


def test_stellung_im_verteilung_ist_p_tau_kleiner_tau():
    surv = np.array([1.0, 1.0, 0.5, 0.25, 0.0])
    assert ev.position_in_distribution(surv, 3) == 0.5                  # P(τ < 3) = P(τ ≤ 2) = 1 − surv[2]
    assert ev.position_in_distribution(surv, 1) == 0.0
    assert ev.position_in_distribution(surv, 99) is None and ev.position_in_distribution(None, 3) is None


def test_lauf_ohne_exakte_verteilung_und_ungerades_n():
    kind, text = ev.verdict_lauf(_res("ok", n=12, run=_run(40), exact=None, peak=False))
    assert kind == "info" and "keine exakte Verteilung" in text and "**τ = 40 Schritten**" in text
    kind, text = ev.verdict_lauf(_res("ok", n=7, run=_run(40), exact=None, peak=True))
    assert "ungeradem n" in text and "Plätzen 0 bis n − 2" in text


def test_lauf_ohne_treffen_metropolis_lambda_1_nennt_den_beweis_und_andere_nicht():
    kind, text = ev.verdict_lauf(_res("nie", coupling="M", lam=1.0, run=_run(0, steps=300000), exact={"never": True}))
    assert kind == "warning" and "kein Treffen von diesem Startpaar aus" in text and "nach 300 000 Schritten" in text and "mischt trotzdem" in text
    kind, text = ev.verdict_lauf(_res("ohne_treffen", coupling="M", lam=1.0, run=_run(None, steps=5)))
    assert kind == "warning" and "Beweis in Experiment 3" in text and "nach 5 Schritten" in text
    kind, text = ev.verdict_lauf(_res("ohne_treffen", coupling="C", lam=1.0, run=_run(None, steps=5)))
    assert "Beweis" not in text and "Schrittbudget reicht hier nicht" in text


def _cell():
    return {"t_mix": 153, "t_mix_eps": {"0.25": 153, "0.01": 748}, "E_tau": 265.32}


def test_schranke_nennt_beide_zaehleinheiten_und_die_faktoren():
    r = ev.E2Result(8, 5.0, 0.25, _cell(), 153, 280, 209, 346, 265.32, 100, 100, (1, 2), 3, 0)
    kind, text = ev.verdict_schranke(r)
    assert kind == "success"
    assert "t_mix(1/4) = 153 (Zähleinheit d), t̄_mix(1/4) = 280 und T_mix(1/e) = 209 (Zähleinheit d̄)" in text
    assert "t_c(1/4) = 346: **2,26-mal t_mix**, 1,24-mal t̄_mix" in text
    assert "In allen 24 vorgerechneten Zellen (ε = 1/4) liegt t_c beim 1,89- bis 2,28-fachen" in text
    assert "4·E[τ] = 1061" in text and "6,94-fachen" in text and "feste Zeit" not in text
    r2 = ev.E2Result(8, 5.0, 0.01, _cell(), 748, 877, 209, 943, 265.32, 100, 100, (1, 2), 3, 0)
    text2 = ev.verdict_schranke(r2)[1]
    assert "t_mix(0,01) = 748" in text2 and "1,26-mal t_mix" in text2 and "1,08-mal t̄_mix" in text2 and "feste Zeit" in text2


def test_kopplungen_verdikt_nennt_schranke_faktor_und_nie():
    rows = [
        {"coupling": "S", "t_mix": 31, "never": False, "t_c": 66, "factor": 66 / 31},
        {"coupling": "I", "t_mix": 31, "never": False, "t_c": 208, "factor": 208 / 31},
        {"coupling": "M", "t_mix": 15, "never": True, "t_c": None, "factor": None},
    ]
    kind, text = ev.verdict_kopplungen(8, 1.0, rows)
    assert kind == "info" and "t_mix(1/4) = 31 der Heat-bath-Kette" in text
    assert "gemeinsamer Platz und Münze (S): 66 (2,13-fach)" in text and "vollständig unabhängig (I): 208 (6,71-fach)" in text
    assert "Metropolis mit gemeinsamem u (M): **trifft sich nie** (obwohl t_mix der Metropolis-Kette 15 ist)" in text
    assert ev.verdict_kopplungen(8, 1.0, [])[1] == "Mindestens eine Kopplung wählen."
    only_m = ev.verdict_kopplungen(8, 1.0, rows[2:])[1]
    assert "der Heat-bath-Kette" not in only_m


def test_startpaar_verdikt_nennt_die_zellzahlen():
    kind, text = ev.verdict_startpaare()
    assert kind == "info" and "allen 30 gemessenen Zellen" in text and "24 Zellen Kopplung S bei geradem n, 4 Zellen S bei ungeradem n, 2 Zellen Metropolis bei λ = 5" in text


def _e4(row, mean=1166.0, se=56.5, q75=1458, ci=(1275, 1652), runs=100, asked=100, n=24, lam=5.0):
    return ev.E4Result(n, lam, row, asked, runs, (1,), 1, mean, se, q75, ci)


def test_matrixfrei_verdikt_mit_exakter_mischzeit_und_ohne():
    row = {"runs": 2000, "q75": 1564, "E_tau": 1252.8, "se": 14.3, "t_mix": 705, "t_mix_basis": "Start alle geraden Plätze (Untergrenze)"}
    kind, text = ev.verdict_matrixfrei(_e4(row))
    assert kind == "info" and "n = 24, λ = 5 (103 682 Zustände)" in text and "100 Läufe, E[τ] = 1166,0 ± 56,5, 75-%-Quantil 1458 (95-%-Band 1275 bis 1652)" in text
    assert "Vorgerechnet aus 2000 Läufen: q75 = 1564, E[τ] = 1252,8 ± 14,3" in text
    assert "exakte Mischzeit ist 705 (Untergrenze, aus dem Start der geraden Plätze): q75 liegt beim 2,22-fachen" in text and "Stichprobe" not in text
    row40 = {"runs": 1000, "q75": 3091, "E_tau": 2538.1, "se": 35.0, "t_mix": None, "t_mix_basis": None}
    kind, text = ev.verdict_matrixfrei(_e4(row40, n=40, lam=1.0, runs=60, asked=100))
    assert "(von 100 gewünschten; das Schrittbudget begrenzt)" in text and "Mischzeit ist hier nicht bekannt" in text
    assert "mindestens 0,9996" in text                                  # λ = 1: Stichprobe von 10⁵ Ziehungen bei 228 826 127 Zuständen
    assert "keine Messung vor" in ev.verdict_matrixfrei(_e4(None, n=32, lam=20.0))[1]


def test_untergrenze_der_stichproben_tv():
    assert ev.tv_lower_bound_sample(100, 40) == 0.6
    assert ev.tv_lower_bound_sample(100, 400) == 0.0
    assert round(ev.tv_lower_bound_sample(228_826_127, 10**5), 4) == 0.9996
    assert round(ev.tv_lower_bound_sample(103_682, 10_000), 2) == 0.90


def test_laeufe_verdikt():
    r = ev.LaeufeResult(10, 1000, 1805, 3967, (2261, 6499), 0.0102, 0.5254, (3000,), 10)
    kind, text = ev.verdict_laeufe(r)
    assert kind == "info" and "R = 10 Läufen" in text and "von 2261 bis 6499" in text and "Wahrscheinlichkeit 1,02 %" in text and "10 von 1000" in text
    assert "mehr als das Doppelte der Mischzeit 1805" in text
    assert ev.wk_text(1e-7) == "unter 0,01 %" and ev.wk_text(0.0001) == "0,01 %"


def test_satz5_verdikt_nennt_den_drei_zustands_fall():
    kind, text = ev.verdict_satz5()
    assert kind == "info" and "P(T > 1)/ℓ = 1/2" in text and "über d(1) = 1/3" in text and "gleich d̄(1) = 1/2" in text and "0 von 1379 Stützstellen" in text


def test_kopplungsnamen_und_schnellstart_sind_vollstaendig():
    assert set(C.COUPLING_NAMES) == set(C.COUPLING_SHORT) == set(C.COUPLINGS)
