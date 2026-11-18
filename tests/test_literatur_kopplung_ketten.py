"""Literaturwerte (ORACLE-PLAYBOOK §5.1): nur, was in den Quellenberichten als gelesen geführt ist. Stand der Einsicht: 2026-11-18 (Quellenberichte zu Levin/Peres/Wilmer und Propp/Wilson).

* Levin/Peres/Wilmer, *Markov Chains and Mixing Times*, 2. Auflage (gelesen): **Korollar 5.5 (Seite 62)**: d(t) ≤ d̄(t) ≤ max_{x,y} P_{x,y}(τ > t) für jede Kopplung, und
  t_mix ≤ 4·max_{x,y} E_{x,y}[τ] (Markov-Ungleichung). **Kapitel 5, Hyperwürfel:** die Kopplung „gleiche Koordinate, gleiches Bit“ trifft, wenn alle Koordinaten gewählt wurden
  (Sammler-Problem); E[τ] = m·H_m ist die Erwartung des Sammlers.
* Propp/Wilson 1996, Satz 5: **nur nach dem Quellenbericht** (der Wortlaut des Originals wurde nicht erneut geöffnet): P(T*>k)/ℓ ≤ d̄(k) ≤ P(T*>k) mit ℓ als Länge der längsten Kette der Ordnung.
  Der Test prüft die Aussage als Eigenschaft im Modell, nicht den Wortlaut.

Kein Test aus nicht gelesenen Quellen (Biswas/Jacob/Vanetti-Formel, Sinclair/Jerrum, Norris, Robert/Casella)."""

import math

import numpy as np

import kpl_chain as K
import kpl_evaluation as ev
import kpl_exact as E
import kpl_pairs as P
import kpl_reference as REF


def test_lpw_folgerung_5_5_in_allen_vorgerechneten_zellen():
    for c in REF.curves()["zellen"].values():
        assert c["t_mix"] <= c["t_c_eps"]["0.25"]                       # d(t) ≤ max P(τ > t) ⇒ t_mix ≤ t_c
        assert c["t_mix"] <= 4 * c["E_tau_max"]                        # t_mix ≤ 4 max E[τ]


def test_lpw_folgerung_5_5_fuer_alle_vier_kopplungen_exakt():
    for coupling in ("S", "C", "I", "M"):
        ring = K.get_ring(6)
        lam = 5.0
        kind = "metropolis" if coupling == "M" else "heatbath"
        spec = E.Spectrum(ring.matrix(kind, lam), ring.pi(lam))
        pc = P.PairChain(ring, coupling, lam)
        mx, _ = pc.curve(1500, stop=0.0)
        for t in (1, 5, 20, 60, 150, 400):
            d, db = spec.d_dbar(t)
            assert d <= db + 1e-12 and db <= mx[t] + 1e-12              # d ≤ d̄ ≤ max P(τ > t)


def test_hyperwuerfel_sammler_erwartung_und_ueberlebenskurve():
    for m in (3, 4, 6, 12, 50):
        assert math.isclose(E.hypercube_row(m)["E_tau"], m * sum(1 / i for i in range(1, m + 1)))
    s = E.collector_survival(6, 2000)
    assert math.isclose(s[:-1].sum(), 6 * sum(1 / i for i in range(1, 7)), rel_tol=1e-12)   # E[τ] = Σ_t P(τ > t)


def test_satz_5_nach_quellenbericht_als_eigenschaft_am_ring_und_an_drei_zustaenden():
    st = ev.satz5_stats()
    assert st["left"] == 0 and st["right"] == 0 and st["points"] > 1000
    ts = ev.three_state()
    assert all(r["tail_over_ell"] == r["dbar"] <= r["tail"] for r in ts["rows"])      # hier gilt links sogar Gleichheit
    assert not all(r["tail_over_ell"] <= r["d"] for r in ts["rows"])                  # gegen d statt d̄ geprüft sähe Satz 5 verletzt aus
