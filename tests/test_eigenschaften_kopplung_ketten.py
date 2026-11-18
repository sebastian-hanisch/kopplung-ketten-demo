"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für Kette und Kopplung gelten müssen, ohne dass man das richtige Ergebnis kennt.

Jede Eigenschaft ist eine belegte Aussage (Quelle oder Beweis im Kommentar) und läuft über das ganze Raster n = 4, 6, 8 (und 10, wo billig) mit λ = 0,5 bis 20 oder über feste Zufalls-Seeds."""

import numpy as np
import pytest

import kpl_chain as K
import kpl_exact as E
import kpl_pairs as P
import kpl_reference as REF
import kpl_sim as S

GRID_N = (4, 6, 8)
LAMS = (0.5, 1.0, 2.0, 5.0)


def _chain(n, lam, coupling, T=400):
    ring = K.get_ring(n)
    pc = P.PairChain(ring, coupling, lam)
    st = pc.pair(ring.index(ring.top), ring.index(ring.bot))
    mx, ex = pc.curve(T, ext=st, stop=0.0)
    return ring, pc, st, mx, ex


@pytest.mark.parametrize("coupling", ["S", "C", "I", "M"])
def test_ueberlebenskurve_faellt_und_zeilensummen_von_q_sind_hoechstens_eins(coupling):
    for n in (4, 6):
        for lam in LAMS:
            _, pc, _, mx, ex = _chain(n, lam, coupling, T=200)
            assert np.all(np.diff(ex) <= 1e-15) and np.all(np.diff(mx) <= 1e-15) and ex[0] == 1.0
            assert np.asarray(pc.Q.sum(axis=1)).max() <= 1 + 1e-12 and pc.Q.min() >= 0        # P(τ > t+1) ≤ P(τ > t), substochastisch


@pytest.mark.parametrize("coupling", ["S", "C", "M"])
def test_gemeinsamer_platz_aendert_den_abstand_um_hoechstens_einen_platz(coupling):
    """Bei gemeinsamem Platz ändert ein Schritt höchstens einen Platz je Kette: vom Gipfelpaar (Abstand n) aus ist τ ≥ n, also P(τ > t) = 1 für t < n."""
    for n in GRID_N:
        for lam in LAMS:
            _, _, _, _, ex = _chain(n, lam, coupling, T=n + 2)
            assert np.allclose(ex[:n], 1.0, atol=1e-12)
            if not (coupling == "M" and lam == 1.0):               # Metropolis bei λ = 1 trifft sich nie: die Kurve bleibt bei 1
                assert ex[n] < 1.0


def test_s_und_m_halten_gleiche_zustaende_zusammen_c_nicht():
    """Nach dem Treffen laufen die Kopien gleich weiter: S und M ordnen gleichen Zuständen gleiche Nachfolger zu (gemeinsame Aktualisierungsfunktion); bei C gilt das nur per Vorschrift."""
    ring = K.get_ring(6)
    for c in ("S", "M"):
        for pr, fx, fy in K.kernels(ring, c, 2.0):
            assert np.array_equal(fx, fy)
    assert any(not np.array_equal(fx, fy) for _, fx, fy in K.kernels(ring, "C", 2.0))


def test_schranken_kette_der_ungleichungen_ueber_das_ganze_raster():
    """d ≤ d̄ ≤ 2d; d ≤ P(τ > t); P/ℓ ≤ d̄ ≤ P (Satz 5 nach Quellenbericht); t_mix ≤ t̄_mix ≤ t_c; t_mix ≤ 4·max E[τ]: alle 24 vorgerechneten Zellen."""
    for c in REF.curves()["zellen"].values():
        for s, d, db in zip(c["surv"], c["d"], c["dbar"]):
            assert d <= db + 1e-9 and db <= 2 * d + 1e-9 and d <= s + 1e-9
            assert s / c["ell"] <= db + 1e-9 and db <= s + 1e-9
        assert c["t_mix"] <= c["t_bar"] <= c["t_c_eps"]["0.25"] and c["t_mix"] <= 4 * c["E_tau_max"]
        for e in ("0.25", "0.1", "0.01", "0.001"):
            assert c["t_mix_eps"][e] <= c["t_bar_eps"][e] <= c["t_c_eps"][e]
        assert c["t_c_eps"]["0.25"] <= c["t_c_eps"]["0.1"] <= c["t_c_eps"]["0.01"] <= c["t_c_eps"]["0.001"]


def test_maximum_ueber_alle_paare_gleich_gipfelpaar_bei_s_gerades_n():
    """Monotonie (Levin/Peres/Wilmer, Kapitel 22): bei Kopplung S und geradem n ist das Gipfelpaar das schlechteste Paar, exakt gerechnet in n = 4, 6, 8, λ = 0,5 bis 5."""
    for n in GRID_N:
        for lam in LAMS:
            _, _, _, mx, ex = _chain(n, lam, "S", T=300)
            assert np.abs(mx - ex).max() < 1e-12, (n, lam)


def test_kopplungsschranke_ist_obere_schranke_fuer_alle_vier_kopplungen():
    for coupling in ("S", "C", "I", "M"):
        for n, lam in ((4, 2.0), (6, 5.0), (6, 2.0)):
            ring = K.get_ring(n)
            kind = "metropolis" if coupling == "M" else "heatbath"
            spec = E.Spectrum(ring.matrix(kind, lam), ring.pi(lam))
            t_mix = E.first_t(lambda t: spec.d_dbar(t)[0], 0.25)
            _, pc, st, mx, ex = _chain(n, lam, coupling, T=3000)
            t_c = P.tail_quantile(mx, 0.25)
            assert t_c is not None and t_mix <= t_c
            assert t_mix <= 4 * pc.mean_times().max()


def test_drehung_und_spiegelung_der_plaetze_aendern_die_kurve_nicht():
    n, lam = 6, 2.0
    ring = K.get_ring(n)
    pc = P.PairChain(ring, "S", lam)
    full = (1 << n) - 1

    def rot(m):
        return ((m << 1) | (m >> (n - 1))) & full

    def refl(m):
        return sum(1 << ((n - 1 - v) % n) for v in range(n) if m >> v & 1)

    rng = np.random.default_rng(3)
    for _ in range(12):
        i, j = rng.choice(ring.N, 2, replace=False)
        x, y = int(ring.S[i]), int(ring.S[j])
        base, _ = pc.survival(pc.pair(i, j), t_stop=1e-6, t_cap=200)
        for f in (rot, refl):
            ex, _ = pc.survival(pc.pair(ring.index(f(x)), ring.index(f(y))), t_stop=1e-6, t_cap=200)
            m = min(len(base), len(ex))
            assert np.allclose(base[:m], ex[:m], atol=1e-12)


def test_erwartung_mindestens_n_und_waechst_mit_lambda():
    for c in REF.curves()["zellen"].values():
        assert c["E_tau"] >= c["n"]
    for n in (4, 6, 8, 10):
        es = [REF.cell(n, lam)["E_tau"] for lam in REF.curves()["meta"]["lam_gitter"]]
        ts = [REF.cell(n, lam)["t_mix"] for lam in REF.curves()["meta"]["lam_gitter"]]
        assert es == sorted(es) and ts == sorted(ts)


def test_gleicher_zufalls_seed_gleiche_ausgabe_und_ein_anderer_weicht_ab():
    top, bot = K.peak_masks(8)
    a = S.sample_taus(8, 5.0, "S", 30, 123, top, bot, 10**6)
    b = S.sample_taus(8, 5.0, "S", 30, 123, top, bot, 10**6)
    c = S.sample_taus(8, 5.0, "S", 30, 124, top, bot, 10**6)
    assert a == b and a[0] != c[0]


def test_band_der_schaetzung_wird_mit_r_schmaler_und_die_wahrscheinlichkeit_unter_t_mix_faellt():
    surv = REF.full_curve(10, 20.0)
    c = REF.cell(10, 20.0)
    widths, below = [], []
    for R in (10, 20, 50, 200, 1000):
        lo, hi = E.quantile_estimate_band(surv, R)
        widths.append(hi - lo)
        below.append(E.prob_estimate_below(surv, R, c["t_mix"]))
    assert widths == sorted(widths, reverse=True) and below == sorted(below, reverse=True)
