"""Feste Texte der App (Einleitung, Erklärung, Grenzen, Formeln). Zahlen stehen hier nur, wenn sie in tests/test_claims.py nachgerechnet werden."""

EINLEITUNG = """
Zwei Kopien derselben **Markov-Kette** laufen mit **gemeinsamem Zufall**, bis sie sich treffen. Die **Treffzeit τ** begrenzt die **Mischzeit** der Kette von oben (Korollar 5.5 bei
Levin/Peres/Wilmer: d(t) ≤ P(τ > t)), ohne dass man die Übergangsmatrix aufstellt. Die Demo zeigt das an einem ausgedachten Ring von Stellplätzen (keine zwei benachbarten belegt),
an dem sich die Zahlen bis n = 24 gegen eine exakte Rechnung prüfen lassen (bei n = 40 gibt es nur noch die Läufe): **was die Schranke kostet**, **wie stark sie an der gewählten Kopplung hängt** (eine Kopplung kann leer bleiben, obwohl die
Kette schnell mischt) und **wie weit man ohne Matrix kommt**. **τ ist eine Zeit, keine Probe.**

**Der Ring ist ausgedacht**: ein **Modell, keine reale Datenquelle**. Was die Zahlen für einen echten Betrieb heißen, wird nicht behauptet.

Unten stehen vier Experimente, die **Grenzen der Annahmen** und der Expander **📐 Mathematische Formulierung**.
"""

SO_FUNKTIONIERT = """
1. **Die Kette.** Ein Ring aus n Stellplätzen; ein Zustand ist die Menge der belegten Plätze, keine zwei benachbarten. Je Schritt wird ein Platz gleichverteilt gewählt und eine Münze geworfen
   (Kopf mit Wahrscheinlichkeit λ/(1+λ)): Kopf belegt den Platz, wenn beide Nachbarn leer sind, sonst wird er leer gesetzt (**Heat-bath**, auch Gibbs-Sampler oder Glauber genannt).
2. **Die Kopplung.** Zwei Kopien starten in zwei verschiedenen Zuständen und bekommen **denselben Platz und dieselbe Münze**. Jede Kopie für sich läuft genau die Kette; zusammen
   nähern sie sich, bis sie gleich sind, und laufen dann gleich weiter. Die Zeit bis dahin ist die **Treffzeit τ**.
3. **Warum das eine Schranke gibt.** Startet man die zweite Kopie im Gleichgewicht, bleibt sie dort. Haben sich die Kopien nach t Schritten getroffen, ist die erste gleich der zweiten und damit im
   Gleichgewicht verteilt; nicht getroffen haben sie sich mit Wahrscheinlichkeit P(τ > t). Der Abstand zum Gleichgewicht ist darum höchstens P(τ > t) (das Maximum über alle Startpaare; bei Kopplung S
   reicht das Gipfelpaar). Die Schranke gilt für **diese** Kopplung, nicht für die Kette allgemein.
4. **Was man misst.** Das 75-%-Quantil von τ (Kopplung S, gerades n, Start im Gipfelpaar) ist die Schranke t_c(1/4) für die Mischzeit t_mix(1/4); aus R Läufen erhält man eine Schätzung dieser Schranke. Das Verfahren braucht keine Übergangsmatrix, nur Läufe.
"""

DREI_BEDEUTUNGEN = """
**„Kopplung“ in drei Bedeutungen.**
* **Hier:** zwei Ketten laufen mit gemeinsamem Zufall, bis sie sich treffen; die Zeit bis zum Treffen begrenzt die Mischzeit (Beweis- und Messbegriff).
* **Ziehen aus der Vergangenheit:** dieselbe Kopplung rückwärts in der Zeit liefert eine exakt verteilte Ziehung, [`kopplung-vergangenheit-demo`](https://github.com/sebastian-hanisch/kopplung-vergangenheit-demo);
  die Treffzeit τ ist auch dort eine Zeit und keine Probe.
* **Monte Carlo:** „Kopplung“ heißt dort gemeinsame Zufallszahlen für zwei Schätzungen (Spiegelpaare), [`gekoppelte-ziehungen-demo`](https://github.com/sebastian-hanisch/gekoppelte-ziehungen-demo).
  Der Mechanismus (gemeinsamer Platz, gemeinsame Münze) ist derselbe, der Zweck ein anderer: dort sinkt die Varianz einer Differenz, hier fällt die Zeit bis zur Gleichheit. Die Varianz einer
  Differenz bei gemeinsamen gegen unabhängigen Zufallszahlen ist in dieser Demo nicht gemessen.
"""

KONVENTIONEN = r"""
| Symbol | Bedeutung |
|---|---|
| $n$ | Zahl der Stellplätze im Ring |
| Zustand $x$ | Menge der belegten Plätze, keine zwei benachbarten (der Ring ist geschlossen) |
| $\lambda$ | Gewicht je belegtem Platz: $\pi(x) \propto \lambda^{\vert x \vert}$ |
| $N$ | Zahl der Zustände (Lucas-Zahl: 7 bei $n = 4$, 47 bei $n = 8$) |
| Schritt | eine Platzaktualisierung; bei zwei Kopien zählt der gemeinsame Schritt einmal |
| Heat-bath, Metropolis | die zwei Ketten mit Gleichgewicht $\pi$ (Heat-bath: Platz und Münze; Metropolis: Umkehren des Platzes, angenommen mit $\min(1, \pi(\text{neu})/\pi(\text{alt}))$) |
| S, C, I, M | die vier Kopplungen: S gemeinsamer Platz und Münze, C gemeinsamer Platz und unabhängige Münzen, I vollständig unabhängig bis zum Treffen, M Metropolis mit gemeinsamem Platz und gemeinsamem $u$ |
| $\tau$ | Treffzeit: $\min\{t : X_t = Y_t\}$ (eine Zeit, in Schritten, keine Probe) |
| Gipfelpaar | Startzustände „alle geraden Plätze belegt“ und „alle ungeraden belegt“ (größtes und kleinstes Element der Ordnung bei geradem $n$) |
| $d(t)$ | $\max_x \mathrm{TV}(P^t(x,\cdot), \pi)$; **TV-Mischzeit** $t_{\mathrm{mix}}(\varepsilon) = \min\{t : d(t) \le \varepsilon\}$, Standard $\varepsilon = 1/4$ |
| $\bar d(t)$ | $\max_{x,y} \mathrm{TV}(P^t(x,\cdot), P^t(y,\cdot))$; $\bar t_{\mathrm{mix}}(1/4)$ und $T_{\mathrm{mix}}(1/e) = \min\{t : \bar d(t) \le 1/e\}$ |
| $t_c(\varepsilon)$ | Kopplungsschranke: $\min\{t : P(\tau > t) \le \varepsilon\}$, das $(1-\varepsilon)$-Quantil von $\tau$ |
| $\ell$ | Länge der längsten Kette der Ordnung (am Ring $\ell = n$) |
| Läufe $R$ | Zahl der gekoppelten Läufe; jeder Lauf zieht ein $\tau$ |

**Beide Zähleinheiten stehen in jeder Anzeige:** $d$ (Levin/Peres/Wilmer) und $\bar d$ (Propp/Wilson), weil die Kopplungssätze über $\bar d$ laufen.
"""

FORMELN = r"""
**Gegeben** sind $n \ge 4$ und $\lambda > 0$; **gesucht** ist, was die Treffzeit zweier gekoppelter Kopien über die Mischzeit sagt. Der Block gibt das Modell vollständig an (Beweise stehen nicht hier).

**Ring.** Zustandsraum $\Omega = \{x \in \{0,1\}^n : x_v x_{v+1} = 0 \text{ für alle } v\}$ (Indizes modulo $n$), Gleichgewicht $\pi(x) = \lambda^{\vert x \vert} / Z$ mit $Z = \sum_{x \in \Omega} \lambda^{\vert x \vert}$.

**Heat-bath.** Wähle $v$ gleichverteilt aus $\{0, \dots, n-1\}$ und eine Münze $c$ mit $P(c = 1) = \lambda/(1+\lambda)$. Setze $x_v \leftarrow 1$, wenn $c = 1$ und beide Nachbarn von $v$ leer sind, sonst $x_v \leftarrow 0$.
Das ist die Aktualisierungsfunktion $\phi(x; v, c)$, die Übergangsmatrix ist $P(x,y) = \sum_{(v,c) : \phi(x; v, c) = y} P(v)\,P(c)$.
**Metropolis** (Einzelplatz): $v$ gleichverteilt, $u$ gleichverteilt in $[0,1)$; belegen, wenn $x_v = 0$, beide Nachbarn leer und $u < \min(1, \lambda)$; leeren, wenn $x_v = 1$ und $u < \min(1, 1/\lambda)$.

**Kopplung.** Zwei Ketten $(X_t)$, $(Y_t)$, jede für sich genau die Kette. **S:** $(v, c)$ gemeinsam. **C:** $v$ gemeinsam, $c_X, c_Y$ unabhängig. **I:** $(v_X, c_X)$ und $(v_Y, c_Y)$ unabhängig.
**M:** Metropolis mit gemeinsamem $(v, u)$. Nach dem Treffen laufen beide gleich weiter. Treffzeit $\tau = \min\{t \ge 0 : X_t = Y_t\}$.

**Paarkette.** Solange $X_t \ne Y_t$ ist $(X_t, Y_t)$ eine Markov-Kette auf den $N(N-1)$ geordneten Paaren; $Q$ sei ihre substochastische Matrix (Übergänge auf die Diagonale fehlen). Dann gilt
$$P_{x,y}(\tau > t) = (Q^t \mathbf 1)_{(x,y)}, \qquad (I - Q)\,m = \mathbf 1 \;\Rightarrow\; m_{(x,y)} = E_{x,y}[\tau].$$

**Abstände und Schranken.** Mit $d$, $\bar d$ wie in der Tabelle der Symbole gilt $d(t) \le \bar d(t) \le 2\,d(t)$. Für jede Kopplung gilt $\bar d(t) \le \max_{x,y} P_{x,y}(\tau > t)$ (Levin/Peres/Wilmer, Korollar 5.5), also
$$d(t) \le \max_{x,y} P_{x,y}(\tau > t), \qquad t_{\mathrm{mix}} \le 4 \max_{x,y} E_{x,y}[\tau].$$
Das Maximum über alle Paare wird im Experiment 3 gerechnet; für Kopplung S (gerades $n$, monoton) ist es das Gipfelpaar.
**Satz 5** (Propp/Wilson, nach dem Quellenbericht): mit $\ell$ als Länge der längsten Kette der Ordnung gilt $P(\tau > k)/\ell \le \bar d(k) \le P(\tau > k)$ (monotone Kopplung, Start im größten und kleinsten Element).

**Kopplungsschranke.** $t_c(\varepsilon) = \min\{t : P(\tau_{\text{Gipfelpaar}} > t) \le \varepsilon\}$; die Schranke gilt nur für diese Kopplung.

**Quantil aus $R$ Läufen.** Das Stichprobenquantil $\hat q_R$ ist die $\lceil 0{,}75\,R \rceil$-te kleinste von $R$ unabhängigen Treffzeiten; mit $F(t) = 1 - P(\tau > t)$ ist
$$P(\hat q_R \le t) = P\big(\mathrm{Bin}(R, F(t)) \ge \lceil 0{,}75\,R \rceil\big).$$

**Kontrollfall Hyperwürfel** $\{0,1\}^m$: Koordinate gleichverteilt wählen, auf ein faires Bit setzen; Kopplung: gleiche Koordinate, gleiches Bit. Vom Paar (alle 0, alle 1) ist $\tau$ die Zeit, bis alle $m$ Koordinaten gewählt sind (Sammler):
$$P(\tau \le t) = \sum_{j=0}^{m} (-1)^j \binom{m}{j}\Big(1 - \frac{j}{m}\Big)^t, \qquad E[\tau] = m \sum_{i=1}^{m} \frac1i.$$

**Quellen** (nur so zitiert, wie sie gelesen sind). Levin/Peres/Wilmer, *Markov Chains and Mixing Times*, 2. Auflage: Kapitel 5 (Korollar 5.5, Seite 62; Hyperwürfel mit Sammler-Problem), Kapitel 22 und Abschnitt 25.4 (monotone Kopplung, Hard-Core-Modell).
Propp/Wilson 1996: Satz 5 nach dem Quellenbericht (Wortlaut des Originals nicht erneut geöffnet). Jacob/O'Leary/Atchadé 2020: Kopplung verbessert das Mischen der Einzelketten nicht; Treffzeiten sind Schätzungen aus Wiederholungen.
Biswas/Jacob/Vanetti 2019: berechenbare Schranken aus Kopplungen mit Zeitversatz (in dieser Demo nicht benutzt). Nicht gelesen und nicht zitiert: Norris, das Handbook of Markov Chain Monte Carlo (Buch), Robert/Casella, Sinclair/Jerrum 1989.
Alle Zahlen sind eigene Nachrechnung in diesem Modell. Nichts davon ist inhaltlich neu; neu ist nur die Vorführbarkeit mit Zähleinheit und exakten Orakeln.
"""

METROPOLIS_BEWEIS = r"""
**Warum Metropolis mit gemeinsamem $u$ bei $\lambda = 1$ nie trifft (Beweis in drei Zeilen, Eigenherleitung, exakt bestätigt).**
1. Nach einem Schritt am Platz $v$ können $x'$ und $y'$ nur gleich sein, wenn $x$ und $y$ vorher **nur an $v$** verschieden waren (alle anderen Plätze bleiben unverändert), etwa $x_v = 1$, $y_v = 0$. Dann sind die Nachbarn von $v$ in beiden Zuständen leer.
2. Bei $\lambda = 1$ werden Entfernen und Belegen beide mit Wahrscheinlichkeit $\min(1, \lambda) = \min(1, 1/\lambda) = 1$ angenommen: $x'_v = 0$ und $y'_v = 1$, für jedes $u$. Sie bleiben verschieden.
3. Bei $\lambda \ne 1$ gibt es ein $u$-Intervall, in dem nur eine der beiden Änderungen angenommen wird; für Paare, die sich nur an einem Platz unterscheiden, treffen sie sich dort.

Exakte Bestätigung: die Zeilensummen von $Q$ sind bei **jedem** der $N(N-1)$ Paare gleich 1 (n = 4, 6, 8, 10), obwohl die Metropolis-Kette in endlich vielen Schritten mischt. Das gilt für diese Kopplung bei $\lambda = 1$; nicht gemessen sind
Metropolis bei $\lambda = 2$ und $10$ und ein anderes Zufallsschema für $u$.
"""

SATZ5_ERKLAERUNG = """
**Was Satz 5 sagt, und die Falle d gegen d̄.** Die Aussage P(τ > k)/ℓ ≤ d̄(k) ≤ P(τ > k) läuft über **d̄**, nicht über **d**. Am Ring ist ℓ = n groß, der Bruch klein, und der Unterschied zwischen d und d̄
bleibt unsichtbar. Sichtbar wird er bei kleinem ℓ: in der Kette mit drei Zuständen unten (Kopf: +1, Zahl: −1, am Rand bleiben; ℓ = 2) liegt P(T > 1)/2 über d(1) und gleich d̄(1).
"""

GRENZEN = """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ausgedachtes Modell** (Ring von Stellplätzen) | Es gibt keine reale Datenquelle dafür; die Zahlen gelten nur in diesem Modell. Der Ring ist exakt lösbar, einen Sampler braucht man dort nicht; er dient, weil sich die Aussagen bis n = 24 gegen eine exakte Rechnung prüfen lassen. | [`mischzeit-exakt-demo`](https://github.com/sebastian-hanisch/mischzeit-exakt-demo) (Mischzeit aus der Matrix) |
| **Heat-bath und vier Kopplungen, Start im Gipfelpaar** | Eine andere Kopplung gibt eine andere Schranke (Experiment 3); eine optimierte Kopplung ist nicht gemessen. Die Aussage „die Schranke ist scharf, wenn man richtig koppelt“ steht hier nicht. | in dieser Demo |
| **Gerades n für die Monotonie** | Bei ungeradem n fehlt die Ordnung; gemessen reichen die zwei Startzustände trotzdem (4 Zellen), bewiesen ist es nur mit Monotonie. | in dieser Demo |
| **τ ist eine Zeit, keine Probe** | Hört man beim ersten Treffen vorwärts auf und nimmt den Zustand als Probe, ist sie verzerrt; exakt gezogen wird nur mit der Kopplung aus der Vergangenheit. | [`kopplung-vergangenheit-demo`](https://github.com/sebastian-hanisch/kopplung-vergangenheit-demo) |
| **Nur zwei Modelle** | Der Hard-Core-Ring (1,89- bis 2,28-fach) und der Hyperwürfel (2,10- bis 2,20-fach) geben einen ähnlichen Faktor für die Schranke; mehr heißt das nicht, Ising-Kreis und Gaußmischung sind nicht gemessen. | [`posterior-sampling-demo`](https://github.com/sebastian-hanisch/posterior-sampling-demo) (stetiger Raum, Mehrketten-R̂) |
| **Wenige Läufe, zwei scharfe Gipfel** | Aus zu wenigen Läufen geschätzt, kann die Schranke zu klein sein; bei λ = 100 (zwei scharfe Gipfel) ist das nicht gemessen. | in dieser Demo (Experiment 4) |
| **Zeitversetzte Kopplung** | Schranken für d(t) selbst ohne Matrix (Biswas u. a.) sind hier nicht umgesetzt. | – |
"""
