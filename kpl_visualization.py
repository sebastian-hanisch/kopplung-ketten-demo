"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import kpl_constants as C
import kpl_format as F


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      legend=dict(orientation="h", y=-0.22))
    return lock_axes(fig)


def _vline(fig, x, text, color, dash="dash", y=1.0, anchor="bottom"):
    fig.add_vline(x=x, line=dict(color=color, width=1.5, dash=dash))
    fig.add_annotation(x=x, y=y, yref="paper", text=text, showarrow=False, yanchor=anchor, xanchor="left", font=dict(size=11, color=color), bgcolor="rgba(255,255,255,0.6)")


# --- Experiment 1 -------------------------------------------------------------------------------------------------------------------

def build_streifen(xs, ys, n: int, upto: int):
    """Beide Ketten als Streifenbild (Plätze × Zeit bis Schritt `upto`): oben Kette X, unten Kette Y; belegt = farbig."""
    T = min(upto, len(xs) - 1)
    cols = T + 1
    z = np.full((2 * n + 1, cols), np.nan)
    for t in range(cols):
        x, y = xs[t], ys[t]
        for v in range(n):
            z[v, t] = 1 if x >> v & 1 else 0
            z[n + 1 + v, t] = 2 if y >> v & 1 else 0
    labels = [f"X · {v}" for v in range(n)] + [""] + [f"Y · {v}" for v in range(n)]
    fig = go.Figure(go.Heatmap(z=z, x=list(range(cols)), y=labels, zmin=0, zmax=2, showscale=False, xgap=0, ygap=1,
                               colorscale=[[0, "#eef0f2"], [0.25, "#eef0f2"], [0.25, C.COLOR_X], [0.75, C.COLOR_X], [0.75, C.COLOR_Y], [1, C.COLOR_Y]],
                               hovertemplate="Schritt %{x}, %{y}<extra></extra>"))
    fig.update_xaxes(title_text="Schritt t")
    fig.update_yaxes(autorange="reversed", type="category", showticklabels=n <= 12)
    return _base(fig, height=120 + 18 * (2 * n + 1) if n <= 12 else 360, legend=False)


def build_abstand(xs, ys, n: int, upto: int, tau):
    """Zahl der Plätze, an denen sich die beiden Ketten unterscheiden, je Schritt."""
    T = min(upto, len(xs) - 1)
    d = [bin(xs[t] ^ ys[t]).count("1") for t in range(T + 1)]
    fig = go.Figure(go.Scatter(x=list(range(T + 1)), y=d, mode="lines", line=dict(color=C.COLOR_D, width=2, shape="hv"), name="Plätze mit Unterschied",
                               hovertemplate="Schritt %{x}: %{y}<extra></extra>"))
    if tau is not None and tau <= T:
        _vline(fig, tau, f"τ = {F.ganz(tau)}", C.COLOR_BAD)
    fig.update_xaxes(title_text="Schritt t")
    fig.update_yaxes(title_text="Plätze, an denen X und Y sich unterscheiden", range=[0, n])
    return _base(fig, height=240, legend=False)


def build_tau_in_verteilung(surv, tau):
    """P(τ > t) exakt; senkrechte Marke beim τ dieses Laufs."""
    t = np.arange(len(surv))
    fig = go.Figure(go.Scatter(x=t, y=surv, mode="lines", line=dict(color=C.COLOR_SURV, width=2), name="P(τ > t), exakt", hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"))
    _vline(fig, tau, f"dieser Lauf: τ = {F.ganz(tau)}", C.COLOR_BAD)
    fig.update_xaxes(title_text="Schritt t")
    fig.update_yaxes(title_text="P(τ > t)", range=[0, 1.02])
    return _base(fig, height=260, legend=False)


# --- Experiment 2 -------------------------------------------------------------------------------------------------------------------

def build_schranke(r):
    """P(τ > t), d(t), d̄(t) exakt (vorgerechnet) und die Treppe der R Läufe; senkrechte Marken t_mix, t̄_mix, T_mix(1/e), t_c; waagerechte Marke bei ε."""
    c = r.cell
    g = [0] + c["grid"]
    xmax = 1.12 * max(r.t_c, r.t_bar, r.T_mix_e, r.t_mix)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g, y=[1.0] + c["surv"], mode="lines", line=dict(color=C.COLOR_SURV, width=3), name="P(τ > t) der Kopplung (exakt)", hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=g, y=[1.0] + c["dbar"], mode="lines", line=dict(color=C.COLOR_DBAR, width=2, dash="dot"), name="d̄(t)", hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=g, y=[1.0] + c["d"], mode="lines", line=dict(color=C.COLOR_D, width=2), name="d(t)", hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"))
    if r.taus:
        ts = np.sort(np.array(r.taus))
        grid = np.unique(np.concatenate([[0], ts]))
        emp = [(ts > t).mean() for t in grid]
        fig.add_trace(go.Scatter(x=grid, y=emp, mode="lines", line=dict(color=C.COLOR_Y, width=1.5, shape="hv"), name=f"Treppe aus {F.ganz(r.runs_done)} Läufen", hovertemplate="t = %{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=r.eps, line=dict(color="gray", width=1, dash="dot"))
    for x, txt, col, y in ((r.t_mix, "t_mix", C.COLOR_D, 1.0), (r.t_bar, "t̄_mix", C.COLOR_DBAR, 0.9), (r.t_c, "t_c", C.COLOR_SURV, 0.8)):
        _vline(fig, x, f"{txt} = {F.ganz(x)}", col, y=y, anchor="top")
    fig.update_xaxes(title_text="Schritt t", range=[0, xmax])
    fig.update_yaxes(title_text="Wert", range=[0, 1.03])
    return _base(fig, height=400)


# --- Experiment 3 -------------------------------------------------------------------------------------------------------------------

def build_kopplungen(rows, t_mix_heat, xmax=None):
    """P(τ > t) (Maximum über die gewählten Startpaare) je Kopplung; Marke: Mischzeit der Heat-bath-Kette."""
    colors = {"S": C.COLOR_SURV, "C": C.COLOR_Y, "I": C.COLOR_DBAR, "M": C.COLOR_BAD}
    fig = go.Figure()
    top = 0
    for r in rows:
        g, cur = r["grid"], r["curve"]
        fig.add_trace(go.Scatter(x=g, y=cur, mode="lines", line=dict(color=colors[r["coupling"]], width=2.5), name=C.COUPLING_SHORT[r["coupling"]],
                                 hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"))
        top = max(top, g[-1] if r["never"] else (r["t_c"] or g[-1]) * 1.4)
    fig.add_hline(y=0.25, line=dict(color="gray", width=1, dash="dot"))
    if t_mix_heat:
        _vline(fig, t_mix_heat, f"t_mix = {F.ganz(t_mix_heat)}", C.COLOR_D, y=1.0, anchor="top")
    fig.update_xaxes(title_text="Schritt t", range=[0, xmax or top])
    fig.update_yaxes(title_text="P(τ > t)", range=[0, 1.03])
    return _base(fig, height=380)


# --- Experiment 4 -------------------------------------------------------------------------------------------------------------------

def build_skalierung(rows, lam):
    """q75(τ) und E[τ] ± Standardfehler gegen n (aus Läufen), dazu die exakte Mischzeit, wo bekannt (vorgerechnet)."""
    rs = [r for r in rows if r["lam"] == lam]
    ns = [r["n"] for r in rs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r["q75"] for r in rs], mode="lines+markers", line=dict(color=C.COLOR_SURV, width=2.5), name="75-%-Quantil von τ (Läufe)",
                             hovertemplate="n = %{x}: %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ns, y=[r["E_tau"] for r in rs], mode="markers", marker=dict(color=C.COLOR_Y, size=8), name="Mittel von τ (± Standardfehler)",
                             error_y=dict(type="data", array=[r["se"] for r in rs], visible=True), hovertemplate="n = %{x}: %{y:.1f}<extra></extra>"))
    known = [r for r in rs if r["t_mix"] is not None]
    fig.add_trace(go.Scatter(x=[r["n"] for r in known], y=[r["t_mix"] for r in known], mode="lines+markers", line=dict(color=C.COLOR_D, width=2), name="exakte Mischzeit t_mix",
                             hovertemplate="n = %{x}: %{y}<extra></extra>"))
    fig.update_xaxes(title_text="Plätze n", tickvals=ns)
    fig.update_yaxes(title_text="Schritte")
    return _base(fig, height=360)


def build_laeufe(res, surv):
    """Wiederholte Schätzungen des 75-%-Quantils aus je R Läufen (Histogramm); exaktes 5-%-bis-95-%-Band, Marken t_mix und t_c."""
    fig = go.Figure(go.Histogram(x=list(res.estimates), marker_color=C.COLOR_SURV, opacity=0.85, name="Schätzungen", hovertemplate="%{x}: %{y}<extra></extra>"))
    lo, hi = res.band
    fig.add_vrect(x0=lo, x1=hi, fillcolor=C.COLOR_BAND, line_width=0, annotation_text="exaktes 5-%-bis-95-%-Band", annotation_position="top left")
    _vline(fig, res.t_mix, f"t_mix = {F.ganz(res.t_mix)}", C.COLOR_D, y=0.85, anchor="top")
    _vline(fig, res.t_c, f"exaktes q75 = {F.ganz(res.t_c)}", C.COLOR_BAD, y=0.7, anchor="top")
    fig.update_xaxes(title_text="geschätztes 75-%-Quantil von τ (Schritte)")
    fig.update_yaxes(title_text="Zahl der Wiederholungen")
    return _base(fig, height=320, legend=False)
