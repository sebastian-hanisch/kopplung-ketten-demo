"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Schnellstart-Beispiele."""

LAM_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)       # λ: die Werte, für die vorgerechnet wurde (Experiment 1 bis 3 rechnen live, wo es geht)
COUPLINGS = ("S", "C", "I", "M")
COUPLING_NAMES = {
    "S": "S: gemeinsamer Platz und gemeinsame Münze",
    "C": "C: gemeinsamer Platz, unabhängige Münzen",
    "I": "I: beide Ketten unabhängig bis zum Treffen",
    "M": "M: Metropolis, gemeinsamer Platz und gemeinsames u",
}
COUPLING_SHORT = {
    "S": "gemeinsamer Platz und Münze (S)",
    "C": "gemeinsamer Platz, unabhängige Münzen (C)",
    "I": "vollständig unabhängig (I)",
    "M": "Metropolis mit gemeinsamem u (M)",
}

# Experiment 1: zwei Kopien, ein gemeinsamer Zufall
E1_N_MIN, E1_N_MAX, DEFAULT_E1_N = 4, 24, 8
DEFAULT_E1_LAM = 5.0
DEFAULT_E1_KOPPLUNG = "S"
START_ARTEN = ("Gipfelpaar", "zwei zufällige Belegungen", "eigene Eingabe")
DEFAULT_START = START_ARTEN[0]
DEFAULT_E1_X, DEFAULT_E1_Y = "10101010", "01010101"
EXAKT_LIVE_MAX_N = 8               # bis hierhin wird die exakte Verteilung von τ live gerechnet (Paarkette: 2 162 Paare bei n = 8)
EINGEFROREN_N = (4, 6, 8, 10)      # Ringgrößen der vorgerechneten Datei
UNABHAENGIG_MAX_N = 8              # Kopplung I nur bis hierhin (die Treffzeit wächst etwa mit der Zustandszahl)
BILD_MAX_SCHRITTE = 1200           # so viele Schritte zeigt das Streifenbild höchstens

# Rechenbudget (Zähleinheit Schritte; ein gekoppelter Schritt dauert in dieser Umgebung rund 2 µs)
SIM_STEP_BUDGET = 300_000
LAUF_MAX_SCHRITTE = SIM_STEP_BUDGET

# Experiment 2: die Schranke
E2_NS = (4, 6, 8, 10)
DEFAULT_E2_N, DEFAULT_E2_LAM = 8, 5.0
EPS_WERTE = (0.25, 0.1, 0.01, 0.001)
DEFAULT_EPS = 0.25
E2_RUNS = (10, 20, 50, 100, 200, 500, 1000)
DEFAULT_E2_RUNS = 200

# Experiment 3: welche Kopplung
E3_NS = (6, 8)
E3_LAMS = (1.0, 5.0)
DEFAULT_E3_N, DEFAULT_E3_LAM = 8, 5.0
E3_MODI = ("alle Paare", "nur Gipfelpaar")
DEFAULT_E3_MODUS = E3_MODI[0]

# Experiment 4: ohne Matrix
E4_NS = (8, 12, 16, 20, 24, 32, 40)
E4_LAMS = (1.0, 5.0, 20.0)
DEFAULT_E4_N, DEFAULT_E4_LAM = 24, 5.0
E4_RUNS = (10, 20, 50, 100, 200, 500, 1000)
DEFAULT_E4_RUNS = 100
E4B_N, E4B_LAM = 10, 20.0
E4B_RUNS = (10, 20, 50, 200, 1000)
E4B_REPS = (50, 200, 1000)
DEFAULT_E4B_RUNS, DEFAULT_E4B_REPS = 20, 200

SEED_MAX, DEFAULT_SEED = 999999, 6101

# Farben
COLOR_X = "#4c78a8"
COLOR_Y = "#e08a2c"
COLOR_OK = "#2F6B65"
COLOR_BAD = "#B5483A"
COLOR_GREY = "rgba(150,150,150,0.55)"
COLOR_BAND = "rgba(76,120,168,0.18)"
COLOR_SURV = "#4c78a8"
COLOR_D = "#2F6B65"
COLOR_DBAR = "#7b5ea7"

# Schnellstart: jedes Beispiel lädt alle Regler vollständig; was nicht genannt ist, geht auf die Voreinstellung zurück.
PRESET_STANDARD = "Ein Lauf (n = 8, λ = 5)"
PRESETS = {
    PRESET_STANDARD: {},
    "Kleinstfall von Hand (n = 4, λ = 1)": {"e1_n": 4, "e1_lam": 1.0, "e2_n": 4, "e2_lam": 1.0},
    "Die Schranke (n = 8, λ = 5)": {"e2_runs": 1000},
    "Kopplung wählen (n = 8, λ = 1)": {"e3_lam": 1.0},
    "Metropolis bei λ = 1: treffen sich nie": {"e1_kopplung": "M", "e1_lam": 1.0, "e3_lam": 1.0},
    "Großer Ring ohne Matrix (n = 40, λ = 5)": {"e4_n": 40, "e4_lam": 5.0},
    "Zu wenige Läufe (n = 10, λ = 20, R = 10)": {"e4b_runs": 10, "e4b_reps": 1000},
    "Drei Zustände: d gegen d̄": {"e2_satz5": True},
}
PRESET_HELP = {
    PRESET_STANDARD: "Experiment 1: zwei Kopien des Rings mit 8 Plätzen und λ = 5, Start im Gipfelpaar, Kopplung S; ein Lauf mit seiner Treffzeit τ.",
    "Kleinstfall von Hand (n = 4, λ = 1)": "Ring mit 4 Plätzen und 7 Zuständen: kleinstes mögliches τ = 4, Mittel 135/7; derselbe Fall steht in der README von Hand gerechnet.",
    "Die Schranke (n = 8, λ = 5)": "Experiment 2: das 75-%-Quantil der Treffzeit gegen die exakte Mischzeit, mit 1 000 Läufen als Treppe.",
    "Kopplung wählen (n = 8, λ = 1)": "Experiment 3: dieselbe Kette mit vier Kopplungen; die Schranke hängt an der Kopplung.",
    "Metropolis bei λ = 1: treffen sich nie": "Experiment 1 und 3: die Metropolis-Kette mit gemeinsamem Platz und gemeinsamem u trifft sich bei λ = 1 aus keinem Startpaar, obwohl sie schnell mischt.",
    "Großer Ring ohne Matrix (n = 40, λ = 5)": "Experiment 4: 228 826 127 Zustände, keine Matrix; nur die Läufe und das 75-%-Quantil von τ.",
    "Zu wenige Läufe (n = 10, λ = 20, R = 10)": "Experiment 4: wie oft liegt das aus nur 10 Läufen geschätzte 75-%-Quantil unter der exakten Mischzeit?",
    "Drei Zustände: d gegen d̄": "Experiment 2: ein Beispiel mit drei Zuständen, in dem der Vergleich gegen d statt d̄ Satz 5 verletzt aussehen lässt.",
}
