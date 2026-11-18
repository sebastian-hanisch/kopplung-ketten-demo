"""Permalink (Begrenzen), Schnellstart-Beispiele und Zufalls-Seed-Button.

Alle Regler der Experimente stehen in SETTING_SPECS: ein Beispiel ist aktiv, wenn jeder Regler dem Beispiel (sonst der Voreinstellung) entspricht. Ein Beispiel lädt alle Regler vollständig:
was es nicht nennt, geht auf die Voreinstellung zurück. Reine Ansichtsregler (Schritt im Streifenbild) ändern nichts am Ergebnis und stehen nicht in SETTING_SPECS."""

import random
from dataclasses import dataclass

import streamlit as st

import kpl_constants as C


def _bool(s) -> bool:
    return str(s).strip().lower() in ("1", "true", "ja")


def _text(s) -> str:
    return str(s)[:40]


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None


SETTING_SPECS = {
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "e1_n": SettingSpec("n1", int, C.DEFAULT_E1_N, C.E1_N_MIN, C.E1_N_MAX),
    "e1_lam": SettingSpec("l1", float, C.DEFAULT_E1_LAM, options=C.LAM_GRID),
    "e1_kopplung": SettingSpec("k1", str, C.DEFAULT_E1_KOPPLUNG, options=C.COUPLINGS),
    "e1_start": SettingSpec("s1", str, C.DEFAULT_START, options=C.START_ARTEN),
    "e1_x": SettingSpec("x1", _text, C.DEFAULT_E1_X),
    "e1_y": SettingSpec("y1", _text, C.DEFAULT_E1_Y),
    "e2_n": SettingSpec("n2", int, C.DEFAULT_E2_N, options=C.E2_NS),
    "e2_lam": SettingSpec("l2", float, C.DEFAULT_E2_LAM, options=C.LAM_GRID),
    "e2_eps": SettingSpec("e2", float, C.DEFAULT_EPS, options=C.EPS_WERTE),
    "e2_runs": SettingSpec("r2", int, C.DEFAULT_E2_RUNS, options=C.E2_RUNS),
    "e2_satz5": SettingSpec("s5", _bool, False),
    "e3_n": SettingSpec("n3", int, C.DEFAULT_E3_N, options=C.E3_NS),
    "e3_lam": SettingSpec("l3", float, C.DEFAULT_E3_LAM, options=C.E3_LAMS),
    "e3_S": SettingSpec("c3s", _bool, True),
    "e3_C": SettingSpec("c3c", _bool, True),
    "e3_I": SettingSpec("c3i", _bool, True),
    "e3_M": SettingSpec("c3m", _bool, True),
    "e3_modus": SettingSpec("m3", str, C.DEFAULT_E3_MODUS, options=C.E3_MODI),
    "e4_n": SettingSpec("n4", int, C.DEFAULT_E4_N, options=C.E4_NS),
    "e4_lam": SettingSpec("l4", float, C.DEFAULT_E4_LAM, options=C.E4_LAMS),
    "e4_runs": SettingSpec("r4", int, C.DEFAULT_E4_RUNS, options=C.E4_RUNS),
    "e4b_runs": SettingSpec("r4b", int, C.DEFAULT_E4B_RUNS, options=C.E4B_RUNS),
    "e4b_reps": SettingSpec("w4b", int, C.DEFAULT_E4B_REPS, options=C.E4B_REPS),
}
ANSICHT_REGLER = ("e1_schritt",)      # ändern nichts am Ergebnis, stehen nicht in SETTING_SPECS


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _begrenze(spec: SettingSpec, value):
    if spec.options is not None:
        return value if value in spec.options else spec.default
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            st.session_state[state_key] = _begrenze(spec, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Alle Regler auf die Voreinstellung, dann die Werte des Beispiels."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = spec.default
    for state_key, value in C.PRESETS[name].items():
        st.session_state[state_key] = value
    st.session_state["e1_schritt"] = 0


def aktives_preset():
    """Name des Beispiels, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None (zustandslos)."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default)
               for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Beispiel ist hervorgehoben (`type="primary"`)."""
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispiel laden (die übrigen Regler gehen auf die Voreinstellung zurück); das Ergebnis steht beim jeweiligen Experiment:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
