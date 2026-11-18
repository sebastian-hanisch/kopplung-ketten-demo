"""Kopplung: zwei Ketten treffen sich - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Vier Experimente: ein gekoppelter Lauf, die Schranke für die Mischzeit, welche Kopplung und welche Startpunkte, wie weit man ohne Matrix kommt. Aufbau nach DEMO-PLAYBOOK §2
(Schnellstart in der Seitenleiste, Regler beim Experiment, Kernfragen vor dem Expander).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import kpl_chain as K
import kpl_constants as C
import kpl_exact as E
import kpl_evaluation as ev
import kpl_format as F
import kpl_reference as REF
import kpl_texts as T
from kpl_layout import dunkles_design, kopf_css
from kpl_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from kpl_visualization import build_abstand, build_kopplungen, build_laeufe, build_schranke, build_skalierung, build_streifen, build_tau_in_verteilung

st.set_page_config(page_title="Kopplung – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _lauf(settings):
    return ev.analyse_lauf(settings)


@st.cache_data(show_spinner=False)
def _schranke(n, lam, eps, runs, seed):
    return ev.analyse_schranke(n, lam, eps, runs, seed)


@st.cache_data(show_spinner=False)
def _matrixfrei(n, lam, runs, seed):
    return ev.analyse_matrixfrei(n, lam, runs, seed)


@st.cache_data(show_spinner=False)
def _laeufe(runs, reps, seed):
    return ev.analyse_laeufe(runs, reps, seed)


def _lam_fmt(x):
    return F.lam(x)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kpl_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔗 Kopplung: zwei Ketten treffen sich")
st.markdown(T.EINLEITUNG)

with st.expander("So funktioniert die Kopplung", expanded=True):
    st.markdown(T.SO_FUNKTIONIERT)
    st.markdown(T.DREI_BEDEUTUNGEN)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Beispiel ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.caption("Die Regler der Experimente stehen beim jeweiligen Experiment. Hier steht nur der Zufalls-Seed für alle Läufe.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=randomize_seed)
seed = int(seed)

# --- Kernfragen -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown(
    "**1.** Wann treffen sich zwei Kopien? **2.** Was sagt die Treffzeit über die Mischzeit? **3.** Hängt das an der Kopplung und an den Startpunkten? "
    "**4.** Wie groß darf der Ring sein, wie viele Läufe braucht man?"
)

# --- Experiment 1 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Zwei Kopien, ein gemeinsamer Zufall")
st.caption(f"Ring mit n Stellplätzen, Heat-bath mit Gewicht λ je belegtem Platz; ein Lauf bis zum Treffen, höchstens {F.ganz(C.LAUF_MAX_SCHRITTE)} Schritte. Oben Kette X, unten Kette Y; belegt = farbig.")
a1, a2 = st.columns(2)
with a1:
    n1 = st.slider("Plätze n", *bounds("e1_n"), key="e1_n", help="Der Ring ist geschlossen: Platz n − 1 grenzt an Platz 0.")
    lam1 = st.select_slider("Gewicht λ je belegtem Platz", C.LAM_GRID, key="e1_lam", format_func=_lam_fmt, help="Großes λ: der Ring ist meist fast voll belegt, zwei Gipfel (gerade oder ungerade Plätze) werden scharf.")
with a2:
    koppl1 = st.radio("Kopplung", C.COUPLINGS, key="e1_kopplung", format_func=C.COUPLING_NAMES.get,
                      help="Alle vier sind Kopplungen derselben Kette (jede Kopie für sich läuft genau die Kette); M gehört zur Metropolis-Kette.")
    start1 = st.radio("Startpaar", C.START_ARTEN, key="e1_start", horizontal=True,
                      help="Gipfelpaar: alle geraden gegen alle ungeraden Plätze belegt. Zufällige Belegungen kommen aus dem Zufalls-Seed.")
b1, b2 = st.columns(2)
eigen = start1 == "eigene Eingabe"
with b1:
    st.text_input("Kette X (Platz 0 zuerst, 0 = frei, 1 = belegt)", key="e1_x", disabled=not eigen, help="Wirkt nur beim Startpaar „eigene Eingabe“.")
with b2:
    st.text_input("Kette Y", key="e1_y", disabled=not eigen, help="Wirkt nur beim Startpaar „eigene Eingabe“.")
s1 = ev.E1Settings(int(n1), float(lam1), koppl1, start1, seed, str(st.session_state["e1_x"]), str(st.session_state["e1_y"]))
with st.spinner("Lauf wird gerechnet …"):
    r1 = _lauf(s1)
if r1.run is not None and r1.run.xs is not None:
    max_view = min(len(r1.run.xs) - 1, C.BILD_MAX_SCHRITTE)
    if max_view >= 2:
        st.session_state["e1_schritt"] = min(int(st.session_state.get("e1_schritt", 0)), max_view)
        view = st.slider("Gezeigt bis Schritt (0 = ganzer Lauf)", 0, max_view, key="e1_schritt", help="Nur die Ansicht; der Lauf selbst ändert sich nicht.")
        upto = max_view if view == 0 else int(view)
    else:
        upto = max_view
    m1, m2, m3 = st.columns(3)
    m1.metric("Treffzeit τ", F.ganz(r1.run.tau) if r1.run.tau is not None else "kein Treffen", help="Schritte bis zum ersten Zeitpunkt, an dem beide Ketten im selben Zustand sind.")
    m2.metric("Gerechnete Schritte", F.ganz(r1.run.steps))
    m3.metric("Startpaar", "Gipfelpaar" if r1.is_peak else "anderes Paar")
    st.plotly_chart(build_streifen(r1.run.xs, r1.run.ys, int(n1), upto), width="stretch", key="e1_streifen")
    if r1.run.steps > C.BILD_MAX_SCHRITTE:
        st.caption(f"Gezeigt werden höchstens die ersten {F.ganz(C.BILD_MAX_SCHRITTE)} Schritte; der Lauf geht bis Schritt {F.ganz(r1.run.steps)}.")
    st.plotly_chart(build_abstand(r1.run.xs, r1.run.ys, int(n1), upto, r1.run.tau), width="stretch", key="e1_abstand")
    if r1.exact is not None and r1.exact["surv"] is not None and r1.run.tau is not None:
        st.plotly_chart(build_tau_in_verteilung(r1.exact["surv"], r1.run.tau), width="stretch", key="e1_verteilung")
        st.caption(f"Exakte Verteilung von τ für dieses Startpaar ({'vorgerechnet' if r1.exact['source'] == 'vorgerechnet' else 'live aus der Paarkette gerechnet'}).")
art, text = ev.verdict_lauf(r1)
getattr(st, art)(text)

st.markdown("---")

# --- Experiment 2 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Die Schranke: was sagt die Treffzeit über die Mischzeit?")
st.caption("Exakt vorgerechnet (n = 4, 6, 8, 10; Heat-bath, Kopplung S, Start im Gipfelpaar), dazu die Treppe aus Läufen mit dem Zufalls-Seed der Seitenleiste. "
           "Beide Zähleinheiten stehen nebeneinander: d (Abstand zum Gleichgewicht) und d̄ (Abstand zweier Starts).")
c1, c2, c3, c4 = st.columns(4)
with c1:
    n2 = st.select_slider("Plätze n", C.E2_NS, key="e2_n")
with c2:
    lam2 = st.select_slider("Gewicht λ", C.LAM_GRID, key="e2_lam", format_func=_lam_fmt)
with c3:
    eps2 = st.select_slider("Schwelle ε", C.EPS_WERTE, key="e2_eps", format_func=F.eps)
with c4:
    runs2 = st.select_slider("Läufe R", C.E2_RUNS, key="e2_runs", help="Wie viele Läufe die Treppe hat; das Schrittbudget kann sie begrenzen.")
with st.spinner(f"{F.ganz(min(runs2, 1000))} Läufe werden gerechnet (höchstens {F.ganz(C.SIM_STEP_BUDGET)} Schritte) …"):
    r2 = _schranke(int(n2), float(lam2), float(eps2), int(runs2), seed)
k1, k2, k3, k4 = st.columns(4)
k1.metric(f"t_mix({F.eps(r2.eps)}), Zähleinheit d", F.ganz(r2.t_mix))
k2.metric(f"t̄_mix({F.eps(r2.eps)}), Zähleinheit d̄", F.ganz(r2.t_bar))
k3.metric("T_mix(1/e), Zähleinheit d̄", F.ganz(r2.T_mix_e))
k4.metric(f"Kopplung t_c({F.eps(r2.eps)})", F.ganz(r2.t_c), delta=f"{F.faktor(r2.t_c / r2.t_mix)}-mal t_mix", delta_color="off")
st.plotly_chart(build_schranke(r2), width="stretch", key="e2_chart")
tab2 = pd.DataFrame({
    "Größe": [f"t_mix({F.eps(r2.eps)}) aus d", f"t̄_mix({F.eps(r2.eps)}) aus d̄", "T_mix(1/e) aus d̄", f"t_c({F.eps(r2.eps)}) aus der Kopplung", "Mittel E[τ]", "Mittelwert-Schranke 4·E[τ]"],
    "Schritte": [F.ganz(r2.t_mix), F.ganz(r2.t_bar), F.ganz(r2.T_mix_e), F.ganz(r2.t_c), F.dez(r2.E_tau), F.ganz(4 * r2.E_tau)],
    "Verhältnis zu t_mix": [F.faktor(1), F.faktor(r2.t_bar / r2.t_mix), F.faktor(r2.T_mix_e / r2.t_mix), F.faktor(r2.t_c / r2.t_mix), F.faktor(r2.E_tau / r2.cell["t_mix_eps"]["0.25"]),
                            F.faktor(4 * r2.E_tau / r2.cell["t_mix_eps"]["0.25"])],
})
st.dataframe(tab2, hide_index=True, width="stretch")
st.caption("Verhältnisse zu t_mix(ε); bei E[τ] und 4·E[τ] zu t_mix(1/4), weil die Mittelwert-Schranke kein ε hat. " +
           (f"Treppe: {F.ganz(r2.runs_done)} von {F.ganz(r2.runs_asked)} gewünschten Läufen (Schrittbudget); " if r2.runs_done < r2.runs_asked else f"Treppe: {F.ganz(r2.runs_done)} Läufe; ") +
           (f"75-%-Quantil der Läufe {F.ganz(E.sample_quantile(list(r2.taus), 0.75))} gegen exaktes t_c(1/4) = {F.ganz(r2.cell['t_c_eps']['0.25'])}." if r2.taus else ""))
art, text = ev.verdict_schranke(r2)
getattr(st, art)(text)

st.toggle("Satz 5 und das Drei-Zustands-Beispiel zeigen", key="e2_satz5", help="Die Aussage P(τ > k)/ℓ ≤ d̄(k) ≤ P(τ > k) läuft über d̄; das Beispiel zeigt, was beim Prüfen gegen d passiert.")
if st.session_state["e2_satz5"]:
    st.markdown(T.SATZ5_ERKLAERUNG)
    ss = ev.satz5_stats()
    st.markdown(f"**Am Ring** (n = 4, 6, 8, 10; λ = 0,5 bis 20; ℓ = n; {F.ganz(ss['points'])} Stützstellen in t): d(t) ≤ P(τ > t) verletzt in {ss['d_le_P']}, linke Seite von Satz 5 verletzt in {ss['left']}, "
                f"rechte in {ss['right']}, d ≤ d̄ ≤ 2d verletzt in {ss['dbar_2d']}. Gegen d statt d̄ geprüft sähe die linke Seite verletzt aus in {ss['wrong_d']}.")
    ts = ev.three_state()
    tab5 = pd.DataFrame({
        "k": [r["k"] for r in ts["rows"]],
        "P(T > k)": [F.bruch(r["tail"]) for r in ts["rows"]],
        "P(T > k)/ℓ mit ℓ = 2": [F.bruch(r["tail_over_ell"]) for r in ts["rows"]],
        "d(k)": [F.bruch(r["d"]) for r in ts["rows"]],
        "d̄(k)": [F.bruch(r["dbar"]) for r in ts["rows"]],
    })
    st.dataframe(tab5, hide_index=True, width="stretch", height=min(38 + 35 * len(tab5), 400))
    art, text = ev.verdict_satz5()
    getattr(st, art)(text)

with st.expander("🧪 Zweites Modell: der Hyperwürfel (Kontrollfall mit geschlossener Form)"):
    st.markdown("Kette: eine Koordinate gleichverteilt wählen, auf ein faires Bit setzen; Kopplung: gleiche Koordinate, gleiches Bit (Sammler). Alle Zahlen exakt über Zählketten, Zähleinheit Schritte.")
    hc = ev.hypercube_table()
    st.dataframe(pd.DataFrame({
        "Koordinaten m": [r["m"] for r in hc], "t_mix(1/4)": [F.ganz(r["t_mix"]) for r in hc], "t̄_mix(1/4)": [F.ganz(r["t_bar"]) for r in hc], "t_c(1/4)": [F.ganz(r["t_c"]) for r in hc],
        "t_c / t_mix": [F.faktor(r["c_over_mix"]) for r in hc], "t_c / t̄_mix": [F.faktor(r["c_over_bar"]) for r in hc], "E[τ] = m·H_m": [F.dez(r["E_tau"], 1) for r in hc],
    }), hide_index=True, width="stretch")
    st.caption("Ein zweites, ganz anderes Modell mit ähnlichem Faktor; mehr heißt das nicht.")

st.markdown("---")

# --- Experiment 3 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Welche Kopplung, welche Startpunkte?")
st.caption("Dieselbe Heat-bath-Kette mit vier Kopplungen (M gehört zur Metropolis-Kette), exakt vorgerechnet, Maximum über **alle** Startpaare. Die Treffzeit gehört zur Kopplung, nicht zur Kette.")
d1, d2, d3 = st.columns([1, 1, 2])
with d1:
    n3 = st.select_slider("Plätze n", C.E3_NS, key="e3_n")
with d2:
    lam3 = st.select_slider("Gewicht λ", C.E3_LAMS, key="e3_lam", format_func=_lam_fmt)
with d3:
    t1, t2, t3, t4 = st.columns(4)
    sel = {c: w.toggle(c, key=f"e3_{c}", help=C.COUPLING_NAMES[c]) for c, w in zip(C.COUPLINGS, (t1, t2, t3, t4))}
modus3 = st.radio("Startpaare", C.E3_MODI, key="e3_modus", horizontal=True, help="„alle Paare“: das Maximum von P(τ > t) über alle Startpaare; „nur Gipfelpaar“: nur der Start in den beiden Gipfeln.")
chosen = tuple(c for c in C.COUPLINGS if sel[c])
rows3 = ev.kopplung_rows(int(n3), float(lam3), chosen, modus3)
if rows3:
    t_mix_h = next((r["t_mix"] for r in rows3 if r["coupling"] != "M"), None)
    st.plotly_chart(build_kopplungen(rows3, t_mix_h), width="stretch", key="e3_chart")
    tab3 = pd.DataFrame({
        "Kopplung": [C.COUPLING_SHORT[r["coupling"]] for r in rows3],
        "t_mix(1/4) der Kette": [F.ganz(r["t_mix"]) for r in rows3],
        "t_c(1/4)": ["nie" if r["never"] else F.ganz(r["t_c"]) for r in rows3],
        "t_c / t_mix": ["–" if r["never"] else F.faktor(r["factor"]) for r in rows3],
        "größtes E[τ]": ["∞" if r["never"] else F.dez(r["E_max"], 1) for r in rows3],
    })
    st.dataframe(tab3, hide_index=True, width="stretch")
    st.caption("t_mix der Kette: Heat-bath für S, C, I; Metropolis für M (andere Kette, andere Mischzeit).")
art, text = ev.verdict_kopplungen(int(n3), float(lam3), rows3)
getattr(st, art)(text)
art, text = ev.verdict_startpaare()
getattr(st, art)(text)
with st.expander("Größe bei λ = 1: wie viele Schritte braucht welche Kopplung?"):
    gr = ev.groesse_rows()
    st.dataframe(pd.DataFrame({
        "n": [g["n"] for g in gr], "Zustände N": [F.ganz(g["N"]) for g in gr], "t_mix": [F.ganz(g["t_mix"]) for g in gr], "t_c(S)": [F.ganz(g["S"]) for g in gr],
        "t_c(C)": [F.ganz(g["C"]) for g in gr], "t_c(I)": [F.ganz(g["I"]) for g in gr], "t_c(I) / N": [F.faktor(g["I_over_N"]) for g in gr],
    }), hide_index=True, width="stretch")
    st.caption("Maximum über alle Paare, Heat-bath. Die unabhängige Kopplung braucht etwa so viele Schritte, wie der Raum Zustände hat; die gemeinsame wächst mit n: eine Beobachtung an vier Größen, kein Beweis. "
               "Gemeinsamer Platz mit unabhängigen Münzen (C) ist bei λ = 1 schlechter als vollständig unabhängig (I): nur das gemeinsame Paar aus Platz und Münze zahlt sich aus.")
with st.expander("Warum trifft Metropolis bei λ = 1 nie?"):
    st.markdown(T.METROPOLIS_BEWEIS)

st.markdown("---")

# --- Experiment 4 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Ohne Matrix: wie groß darf der Ring sein, wie viele Läufe braucht man?")
st.caption("Heat-bath, Kopplung S, Start im Gipfelpaar. Die Tabelle ist vorgerechnet (Läufe aus SplitMix64 mit festen Zufalls-Seeds); die Läufe darunter rechnet die App live, höchstens "
           f"{F.ganz(C.SIM_STEP_BUDGET)} Schritte.")
e1, e2, e3 = st.columns(3)
with e1:
    n4 = st.select_slider("Plätze n", C.E4_NS, key="e4_n")
with e2:
    lam4 = st.select_slider("Gewicht λ", C.E4_LAMS, key="e4_lam", format_func=_lam_fmt)
with e3:
    runs4 = st.select_slider("Läufe R", C.E4_RUNS, key="e4_runs")
with st.spinner("Läufe werden gerechnet …"):
    r4 = _matrixfrei(int(n4), float(lam4), int(runs4), seed)
if r4.row is not None:
    mm = st.columns(4)
    mm[0].metric("Zustände N", F.ganz(K.lucas(int(n4))))
    mm[1].metric("E[τ] aus diesen Läufen", F.dez(r4.mean, 1))
    mm[2].metric("75-%-Quantil aus diesen Läufen", F.ganz(r4.q75))
    mm[3].metric("exakte Mischzeit", F.ganz(r4.row["t_mix"]) if r4.row["t_mix"] else "nicht bekannt")
rows_all = REF.scaling_rows()
st.plotly_chart(build_skalierung(rows_all, float(lam4)), width="stretch", key="e4_chart")
tab4 = ev.scaling_table()
st.dataframe(pd.DataFrame({
    "λ": [F.lam(r["lam"]) for r in tab4], "n": [r["n"] for r in tab4], "Zustände N": [F.ganz(r["N"]) for r in tab4], "Läufe": [F.ganz(r["runs"]) for r in tab4],
    "E[τ] ± se": [f"{F.dez(r['E_tau'], 1)} ± {F.dez(r['se'], 1)}" for r in tab4], "q75(τ)": [F.ganz(r["q75"]) for r in tab4], "q90(τ)": [F.ganz(r["q90"]) for r in tab4],
    "t_mix (exakt)": [F.ganz(r["t_mix"]) if r["t_mix"] else "–" for r in tab4], "q75 / t_mix": [F.faktor(r["q75_over_mix"]) if r["q75_over_mix"] else "–" for r in tab4],
    "E[τ] / (n ln n)": [F.dez(r["E_over_nln"], 2) for r in tab4], "E[τ] / t_rel": [F.dez(r["E_over_rel"], 2) for r in tab4],
}), hide_index=True, width="stretch", height=min(38 + 35 * len(tab4), 640))
st.caption("Vorgerechnet. t_mix für n ≤ 12 aus allen Startzuständen, für n ≥ 16 nur aus dem Start „alle geraden Plätze belegt“ (Untergrenze der Mischzeit). t_rel = n/g(λ) (Formel der Nachbar-Demo "
           "[`mischzeit-exakt-demo`](https://github.com/sebastian-hanisch/mischzeit-exakt-demo), nur Vergleichsspalte, gerades n).")
art, text = ev.verdict_matrixfrei(r4)
getattr(st, art)(text)

st.markdown("##### Wie viele Läufe braucht die Schranke?")
st.caption(f"Fest: n = {C.E4B_N}, λ = {F.lam(C.E4B_LAM)} (zwei scharfe Gipfel, Kopplung S). Das Band ist exakt (Ordnungsstatistik der exakten Treffzeit-Verteilung); die Wiederholungen darunter werden aus dieser "
           "Verteilung gezogen (nicht neu simuliert: echte Läufe wären zu lang für das Rechenbudget).")
f1, f2 = st.columns(2)
with f1:
    runs4b = st.select_slider("Läufe R je Schätzung", C.E4B_RUNS, key="e4b_runs")
with f2:
    reps4b = st.select_slider("Wiederholungen", C.E4B_REPS, key="e4b_reps")
r4b = _laeufe(int(runs4b), int(reps4b), seed)
st.plotly_chart(build_laeufe(r4b, REF.full_curve(C.E4B_N, C.E4B_LAM)), width="stretch", key="e4b_chart")
art, text = ev.verdict_laeufe(r4b)
getattr(st, art)(text)

st.markdown("---")

# --- Grenzen --------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.GRENZEN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(T.KONVENTIONEN)
    st.markdown(T.FORMELN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [MCMC: von der Kette zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-mcmc.html)."
)
