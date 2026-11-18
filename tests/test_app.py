"""End-to-end Smoke-Test über Streamlits AppTest: lädt app.py mit den Voreinstellungen, klickt jeden Schnellstart-Button, fährt jeden Regler an seine Grenzen und prüft, dass kein
Python-Fehler auftritt (auch `StreamlitDuplicateElementId` bei Diagrammen ohne eindeutiges key= und `StreamlitAPIException` bei Reglern mit berechneten Grenzen)."""

import os
import time

import pytest
from streamlit.testing.v1 import AppTest

import kpl_constants as C
import kpl_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))   # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(m.value for kind in (at.success, at.info, at.warning, at.error) for m in kind)


def test_app_loads_without_exception():
    at = _fresh_app()
    assert len(at.subheader) >= 5                       # vier Experimente und die Grenzen
    assert "Kopplungsschranke t_c(1/4) = 346" in _texte(at)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def _widget(at, key):
    for ort in (at.main, at.sidebar):
        for art in ("slider", "select_slider", "radio", "number_input"):
            try:
                return getattr(ort, art)(key=key)
            except KeyError:
                continue
    return None


def test_every_slider_and_option_at_its_ends_does_not_raise():
    for key, spec in P.SETTING_SPECS.items():
        if spec.options is None and spec.lo is None:
            continue                                        # Schalter und Textfelder: eigene Tests
        for edge in (0, -1):
            at = _fresh_app()
            w = _widget(at, key)
            assert w is not None, key
            value = spec.options[edge] if spec.options is not None else int(spec.lo if edge == 0 else spec.hi)
            w.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("kopplung", C.COUPLINGS)
@pytest.mark.parametrize("n,lam", [(4, 0.5), (8, 20.0), (24, 5.0)])
def test_experiment_1_alle_kopplungen_an_den_raendern(kopplung, n, lam):
    at = _fresh_app()
    at.slider(key="e1_n").set_value(n)
    at.select_slider(key="e1_lam").set_value(lam)
    at.radio(key="e1_kopplung").set_value(kopplung).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert _texte(at)


def test_experiment_1_eigene_eingabe_gueltig_und_ungueltig():
    at = _fresh_app()
    at.radio(key="e1_start").set_value("eigene Eingabe").run(timeout=TIMEOUT)
    at.text_input(key="e1_x").set_value("10101010")
    at.text_input(key="e1_y").set_value("10001000").run(timeout=TIMEOUT)
    assert not at.exception and "τ = " in _texte(at)
    at.text_input(key="e1_y").set_value("11111111").run(timeout=TIMEOUT)
    assert not at.exception and "Kette Y: Zwei benachbarte Plätze sind belegt" in " ".join(e.value for e in at.error)


def test_experiment_1_zufaellige_belegungen_und_gleicher_start():
    at = _fresh_app()
    at.radio(key="e1_start").set_value("zwei zufällige Belegungen").run(timeout=TIMEOUT)
    assert not at.exception
    at.radio(key="e1_start").set_value("eigene Eingabe").run(timeout=TIMEOUT)
    at.text_input(key="e1_x").set_value("10101010")
    at.text_input(key="e1_y").set_value("10101010").run(timeout=TIMEOUT)
    assert not at.exception and "τ = 0" in _texte(at)


def test_experiment_3_ohne_kopplung_und_nur_metropolis():
    at = _fresh_app()
    for c in C.COUPLINGS:
        at.toggle(key=f"e3_{c}").set_value(False)
    at.run(timeout=TIMEOUT)
    assert not at.exception and "Mindestens eine Kopplung wählen." in _texte(at)
    at.toggle(key="e3_M").set_value(True)
    at.select_slider(key="e3_lam").set_value(1.0).run(timeout=TIMEOUT)
    assert not at.exception and "trifft sich nie" in _texte(at)


def test_experiment_4_nicht_gemessene_zeile_gibt_eine_meldung():
    at = _fresh_app()
    at.select_slider(key="e4_lam").set_value(20.0)
    at.select_slider(key="e4_n").set_value(40).run(timeout=TIMEOUT)
    assert not at.exception and "keine Messung vor" in _texte(at)


def test_schrittbudget_haelt_die_rechenzeit_im_rahmen():
    """Das teuerste Szenario je Experiment (Obergrenzen der Läufe) bleibt unter einer großzügigen Zeitgrenze (die Grenze ist ein Timeout, keine Messung)."""
    at = _fresh_app()
    t0 = time.time()
    at.select_slider(key="e2_runs").set_value(1000)
    at.select_slider(key="e2_n").set_value(10)
    at.select_slider(key="e2_lam").set_value(20.0)
    at.select_slider(key="e4_runs").set_value(1000)
    at.select_slider(key="e4_n").set_value(40).run(timeout=60)
    assert not at.exception and time.time() - t0 < 60
