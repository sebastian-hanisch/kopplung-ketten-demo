"""Schnellstart-Beispiele: Vollständigkeit, gültige Werte, Standard gleich Voreinstellung, jedes Beispiel wirkt (verschiedene Einstellungen), Permalink begrenzt."""

import kpl_constants as C
import kpl_presets as P


def test_jedes_beispiel_hat_hilfe_und_nur_bekannte_regler():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert C.PRESET_HELP[name]
        assert set(preset) <= set(P.SETTING_SPECS), name


def test_werte_der_beispiele_liegen_in_den_grenzen_und_optionen():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            elif spec.lo is not None:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_standard_beispiel_ist_die_voreinstellung_und_keine_zwei_beispiele_sind_gleich():
    assert C.PRESETS[C.PRESET_STANDARD] == {} and next(iter(C.PRESETS)) == C.PRESET_STANDARD
    wirksam = [tuple(sorted((k, v) for k, v in p.items() if v != P.SETTING_SPECS[k].default)) for p in C.PRESETS.values()]
    assert len(set(wirksam)) == len(wirksam)
    assert all(w for w in wirksam[1:])                              # jedes Nicht-Standard-Beispiel ändert mindestens einen Regler


def test_voreinstellungen_liegen_in_den_grenzen():
    for key, spec in P.SETTING_SPECS.items():
        if spec.options is not None:
            assert spec.default in spec.options, key
        elif spec.lo is not None:
            assert spec.lo <= spec.default <= spec.hi, key


def test_permalink_begrenzt_auf_grenzen_und_optionen():
    spec = P.SETTING_SPECS["e1_n"]
    assert P._begrenze(spec, 99) == C.E1_N_MAX and P._begrenze(spec, -5) == C.E1_N_MIN
    assert P._begrenze(P.SETTING_SPECS["e1_lam"], 3.0) == C.DEFAULT_E1_LAM        # 3 ist keine Option: Voreinstellung
    assert P._begrenze(P.SETTING_SPECS["e1_kopplung"], "X") == "S"
    assert P._bool("ja") and not P._bool("nein") and len(P._text("x" * 100)) == 40
