# Kopplung: zwei Ketten treffen sich – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kopplung-ketten-demo.streamlit.app/)** -->

Eine von mehreren Demos zu **MCMC: Verteilungen abtasten** (Simulation) für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Zwei Kopien derselben Markov-Kette laufen mit **gemeinsamem Zufall**,
bis sie sich treffen; die **Treffzeit τ** begrenzt die Mischzeit der Kette von oben, ohne dass man die Übergangsmatrix aufstellt. Die Demo schließt an [`mischzeit-exakt-demo`](https://github.com/sebastian-hanisch/mischzeit-exakt-demo)
an, die die Mischzeit **aus der aufgestellten Matrix** rechnet; hier wird sie **aus Läufen** begrenzt. Die nächste Demo, [`kopplung-vergangenheit-demo`](https://github.com/sebastian-hanisch/kopplung-vergangenheit-demo), nutzt
dieselbe Kopplung rückwärts in der Zeit, um exakt zu ziehen. Weitere Nachbarn: [`gibbs-glauber-demo`](https://github.com/sebastian-hanisch/gibbs-glauber-demo) (die Heat-bath-Kette),
[`metropolis-kleiner-raum-demo`](https://github.com/sebastian-hanisch/metropolis-kleiner-raum-demo) (die Metropolis-Kette), [`diagnose-taeuscht-demo`](https://github.com/sebastian-hanisch/diagnose-taeuscht-demo) (Mehrketten-Diagnostik),
[`gekoppelte-ziehungen-demo`](https://github.com/sebastian-hanisch/gekoppelte-ziehungen-demo) (Monte Carlo, wo „Kopplung“ gemeinsame Zufallszahlen heißt), [`posterior-sampling-demo`](https://github.com/sebastian-hanisch/posterior-sampling-demo)
(stetiger Raum), [`markov-queue-demo`](https://github.com/sebastian-hanisch/markov-queue-demo) (Warteschlangen) und [`meinungsbildung-demo`](https://github.com/sebastian-hanisch/meinungsbildung-demo) (Mischzeit auf Graphen, anders gezählt).

**Das Modell ist ausgedacht** und ein **Modell, keine reale Datenquelle**: die Belegung von Stellplätzen im Ring (keine zwei benachbarten belegt). In den gelesenen Quellen gibt es keine reale Datenquelle dafür. Der Ring ist mit einer
Transfermatrix exakt lösbar, einen Sampler braucht man dort nicht; er dient hier, weil sich die Aussagen bis n = 24 gegen eine exakte Rechnung prüfen lassen (bei n = 40 gibt es nur noch die Läufe). Gezeigt wird, was zwei gekoppelte Ketten über die Mischzeit sagen und was nicht;
was die Zahlen für einen echten Betrieb heißen, wird nicht behauptet.

## Warum dieses Problem

Wer eine Markov-Kette als Sampler benutzt, fragt, wie lange sie laufen muss. Die Mischzeit aus der Matrix zu rechnen geht nur, solange die Matrix aufstellbar ist (der Ring hat bei n = 16 genau 2207 Zustände, bei n = 24 genau 103 682,
bei n = 40 genau 228 826 127). Die Kopplung hebt das auf: zwei simulierte Kopien, bis sie sich treffen, Zähleinheit Läufe. **Antwort in diesem Modell** (Heat-bath, Start in den beiden Gipfeln, Zähleinheit Schritte):
das 75-%-Quantil der Treffzeit (für Kopplung S, gerades n, Start im Gipfelpaar) ist eine obere Schranke für die Mischzeit t_mix(1/4), aus R Läufen erhält man eine Schätzung dieser Schranke; bei n = 8, λ = 5 liegt es bei **346 Schritten gegen eine exakte Mischzeit von 153** (2,26-fach), in allen 24 exakt gerechneten Zellen (n = 4, 6, 8, 10; λ = 0,5 bis 20)
beim 1,89- bis 2,28-fachen (Mittel 2,12), aus 2000 bzw. 600 Läufen bei n = 8 bis 24 beim 2,02- bis 2,40-fachen von t_mix (für n ≥ 16 gegen die Untergrenze aus einem Start; Stichprobenquantile, keine exakten Werte). Bei n = 40 (228 826 127 Zustände) bleibt nur eine Schätzung der Schranke: 3091 Schritte bei λ = 5 aus 1000 Läufen.
**Die Zahl gehört aber zur gewählten Kopplung, nicht zur Kette:** bei λ = 1 und n = 8 geben dieselben Heat-bath-Ketten mit gemeinsamem Platz, aber unabhängigen Münzen das 10,58-fache der Mischzeit, vollständig unabhängig das 6,71-fache,
und die Metropolis-Kette mit gemeinsamem u trifft sich bei λ = 1 aus keinem Startpaar je (Beweis in drei Zeilen, exakt bestätigt), obwohl sie in 15 Schritten mischt. **Die Falle ist, die Treffzeit für eine Mischzeit der Kette zu halten.**
Die Zahlen gelten für Heat-bath auf dem Hard-Core-Ring bei den genannten vier Kopplungen, in Schritten. **Lehrbuchwissen** sind die Kopplungsungleichung d(t) ≤ P(τ > t) und der Hyperwürfel; die Zahlen sind eigene Messungen in diesem Modell.
Nichts davon ist inhaltlich neu; neu ist nur die Vorführbarkeit mit Zähleinheit und exakten Orakeln.

## Modell

* **Ring.** n Stellplätze im Kreis; ein Zustand x ist die Menge der belegten Plätze, keine zwei benachbarten (Platz n − 1 grenzt an Platz 0). Gleichgewicht π(x) ∝ λ^|x| (Levin/Peres/Wilmer, Abschnitt 3.3.4). Zahl der Zustände: Lucas-Zahl (n = 4: 7, n = 8: 47, n = 10: 123).
* **Heat-bath** (Gibbs-Sampler, Glauber): ein Platz gleichverteilt, Münze mit Kopf-Wahrscheinlichkeit λ/(1+λ); Kopf belegt den Platz, wenn beide Nachbarn leer sind, sonst wird er leer gesetzt. Ein **Schritt** ist eine Platzaktualisierung; bei zwei Kopien zählt der gemeinsame Schritt einmal.
  **Metropolis** (Einzelplatz): Platz gleichverteilt, Umkehren, angenommen mit Wahrscheinlichkeit min(1, π(neu)/π(alt)); sie dient nur dem Vergleich.
* **Kopplung** heißt hier der Beweis- und Messbegriff: zwei Kopien der Kette, jede für sich genau die Kette, mit gemeinsamer Zufallsquelle bis zum Treffen; danach laufen sie gleich weiter. **Treffzeit** τ = min{t : X_t = Y_t}. Vier Kopplungen:
  **S** gemeinsamer Platz und gemeinsame Münze (für gerades n monoton), **C** gemeinsamer Platz, unabhängige Münzen, **I** beide Ketten vollständig unabhängig bis zum Treffen, **M** Metropolis mit gemeinsamem Platz und gemeinsamem u. In **Monte Carlo** heißt „Kopplung“
  gemeinsame Zufallszahlen (dort sinkt die Varianz einer Differenz, hier fällt die Zeit bis zur Gleichheit); auf dem **Ziehen aus der Vergangenheit** beruht die Nachbar-Demo `kopplung-vergangenheit-demo`.
* **Gipfelpaar:** die Startzustände „alle geraden Plätze belegt“ und „alle ungeraden belegt“ (größtes und kleinstes Element der Ordnung nach Levin/Peres/Wilmer, Beispiel 22.10). Bei ungeradem n gibt es die Ordnung nicht; das Gipfelpaar sind dann die maximalen Belegungen auf den Plätzen 0 bis n − 2.
* **Zwei Zähleinheiten der Mischzeit, in jeder Anzeige:** d(t) = max_x TV(P^t(x,·), π) mit t_mix(1/4) (Levin/Peres/Wilmer) und d̄(t) = max_{x,y} TV(P^t(x,·), P^t(y,·)) mit t̄_mix(1/4) und T_mix(1/e) (Propp/Wilson); es gilt d ≤ d̄ ≤ 2d, und die Kopplungssätze laufen über d̄.
  **Kopplungsschranke** t_c(ε) = min{t : P(τ > t) ≤ ε}, das (1 − ε)-Quantil der Treffzeit des Gipfelpaars.
* Der 📐-Expander der App enthält die Formeln als vollständiges Modell (ohne Beweise) und die Tabelle der Symbole.

## Methodik

* **Paarkette** (`kpl_pairs.py`): solange X_t ≠ Y_t ist das Paar eine Markov-Kette auf den N(N−1) geordneten Paaren mit der substochastischen Matrix Q (die Diagonale ist absorbierend). Dann ist P(τ > t) = Q^t·1 und E[τ] löst (I − Q)m = 1. n = 8 hat 2162 Paare, n = 10 hat 15 006.
  Live gerechnet wird bis n = 8; n = 10 ist vorgerechnet.
* **Abstände und Mischzeiten** (`kpl_exact.py`): d(t) und d̄(t) aus der Spektralzerlegung der symmetrisierten Matrix; t_mix, t̄_mix, T_mix durch Verdoppeln und Halbieren (d und d̄ fallen nicht). Der Ordnungs-Satz von Propp/Wilson (Satz 5, nach dem Quellenbericht) wird als Eigenschaft geprüft.
* **Läufe** (`kpl_sim.py`): nur der ganzzahlige SplitMix64 (`kpl_rng.py`, geprüfte Vorlage, unverändert kopiert; Zufalls-Seed in der Seitenleiste). Läufe laufen nacheinander aus einem Strom: die ersten k Läufe sind für jede Zahl R ≥ k dieselben. Ein Schrittbudget von 300 000 gekoppelten Schritten
  begrenzt die Zahl der Läufe (vorab aus dem Mittel von τ festgelegt, nicht aus den Läufen, damit die gerechneten Läufe unverzerrt bleiben); die App nennt, wie viele Läufe sie gerechnet hat.
* **Vorgerechnet** (`kpl_curves.json`, `kpl_scaling.json`; Skripte `tools/kpl_make_curves.py` und `tools/kpl_make_scaling.py`, je mit `--check`): die exakten Zellen n = 4, 6, 8, 10 mit je sechs Werten von λ (Experiment 2 und 3), die Läufe-Tabelle n = 8 bis 40 (Experiment 4)
  und die vollständige Überlebenskurve für n = 10, λ = 20. App und README kennzeichnen diese Zahlen als „vorgerechnet“.
* **Band der Schätzung** (Experiment 4, zweites Feld): das Stichprobenquantil q̂_R ist die ⌈0,75·R⌉-te kleinste von R Treffzeiten; mit F(t) = 1 − P(τ > t) ist P(q̂_R ≤ t) = P(Binomial(R, F(t)) ≥ ⌈0,75·R⌉). Das Band ist also **exakt** (Ordnungsstatistik der exakten Verteilung), die Wiederholungen darunter werden aus
  dieser Verteilung gezogen (Inversionsverfahren), nicht neu simuliert: echte Läufe wären zu lang für das Rechenbudget.
* **Rechenzeit** (Messung dieser README, nicht im Test; eine CPU, Windows, Python 3.14, gemessen, während andere Rechenprozesse liefen): ein Lauf mit exakter Verteilung bei n ≤ 8 unter 0,1 s (Kopplung S), 1,3 s bei Kopplung I mit λ = 20 (die Paarkette von 2162 Paaren wird live gerechnet);
  die Treppe aus 1000 Läufen bei n = 8, λ = 5 und die 1000 Läufe bei n = 40, λ = 5 je rund 0,5 bis 0,6 s; das Band mit R = 1000 und 1000 Wiederholungen rund 1 s; Metropolis bei λ = 1 läuft bis zur Obergrenze von 300 000 Schritten, rund 0,6 s. Die Streamlit Cloud ist langsamer.

## Befunde (gemessen, keine Behauptungen)

Zähleinheit: Schritte (eine Platzaktualisierung) und Läufe; Heat-bath, Start im Gipfelpaar, wenn nicht anders gesagt. Die Spalte „Test“ nennt den Test in `tests/test_claims.py`, der genau diese Zahl prüft: Ganzzahlen, Brüche und exakte Größen exakt bzw. auf Gleitkomma-Toleranz; Zufallszahlen aus
Läufen mit festem Zufalls-Seed als Band (Abstand unter 4 Standardfehlern gegen die exakte Größe, sonst die eingefrorene Zeile).

| Frage | Befund | Test |
|---|---|---|
| Kleinstfall n = 4, λ = 1 (7 Zustände) von Hand | kleinstes mögliches τ = 4, P(τ > 4) = 63/64, P(τ > 5) = 977/1024, P(τ > 6) = 1871/2048; Mittel E[τ] = 135/7 = 19,29; vom Gipfelpaar aus sind 19 von 42 Paaren erreichbar; t_mix = 12 (d), t̄_mix = 21 und T_mix(1/e) = 16 (d̄), t_c(1/4) = 25 (2,08-fach t_mix, 1,19-fach t̄_mix); 4·E[τ] = 540/7 = 77,1 (6,43-fach) | `test_k1`, `test_handbeispiel.py` |
| Preis der Kopplung, alle 24 Zellen (ε = 1/4) | t_c/t_mix = 1,89 bis 2,28 (Mittel 2,12); t̄_mix/t_mix = 1,43 bis 1,99 (Mittel 1,76); t_c/t̄_mix = 1,00 bis 1,41 (Mittel 1,21); E[τ]/t_mix = 1,45 bis 1,77; 4·E[τ]/t_mix = 5,79 bis 7,07 | `test_k2` |
| Ausgewählte Zellen (n, λ: t_mix / t̄_mix / T_mix(1/e) / t_c; Faktor gegen t_mix) | 4, 20: 671 / 1338 / 966 / 1342 (2,00; gegen t̄_mix 1,00); 6, 1: 21 / 34 / 26 / 44 (2,10); 8, 1: 31 / 48 / 38 / 66 (2,13); **8, 5: 153 / 280 / 209 / 346 (2,26)**; 8, 20: 1403 / 2737 / 1993 / 2983 (2,13); 10, 5: 208 / 364 / 274 / 474 (2,28); 10, 20: 1805 / 3468 / 2539 / 3967 (2,20) | `test_k3` |
| Strengeres ε | n = 8, λ = 5, ε = 1/4 / 0,1 / 0,01 / 0,001: t_mix 153 / 322 / 748 / 1175, t̄_mix 280 / 450 / 877 / 1303, t_c 346 / 516 / 943 / 1369, also t_c/t_mix 2,26 / 1,60 / 1,26 / 1,17; n = 10, λ = 5: 2,28 / 1,65 / 1,29 / 1,18 | `test_k4` |
| Verteilung von τ | n = 8, λ = 5: Quantile 25 / 50 / 75 / 90 / 99 %: 129 / 214 / 346 / 516 / 943 Schritte, Mittel 265,32, kleinstes mögliches τ = 8 (= n); Gipfelpaar ist das schlechteste Startpaar | `test_k5` |
| Zweites Modell (Hyperwürfel, Sammler) | m = 8 / 12 / 50 / 100 / 400: t_mix 12 / 20 / 120 / 275 / 1378, t̄_mix 17 / 28 / 154 / 343 / 1654, t_c 26 / 44 / 257 / 584 / 2893, t_c/t_mix 2,17 / 2,20 / 2,14 / 2,12 / 2,10; E[τ] = m·H_m = 21,7 / 37,2 / 225,0 / 518,7 / 2628,0 | `test_k6` |
| Welche Kopplung? n = 8, λ = 1 (alle Startpaare) | t_mix 31; S 66 (2,13-fach), I 208 (6,71-fach), C 328 (10,58-fach); M: nie (Metropolis-Mischzeit 15) | `test_k7` |
| Vier Zellen (n, λ) | t_c(I)/t_c(S) = 1,95 / 1,24 / 3,15 / 1,52 und t_c(C)/t_c(S) = 2,91 / 1,32 / 4,97 / 1,65 für (6, 1), (6, 5), (8, 1), (8, 5); S hat in allen vier die kleinste Schranke; bei (6, 5): M 219 (2,52-fach der Metropolis-Mischzeit 87), bei (8, 5): M 332 (2,61-fach der Mischzeit 127) | `test_k8` |
| Größe bei λ = 1 (n = 4, 6, 8, 10; N = 7, 18, 47, 123) | t_c(S) = 25 / 44 / 66 / 90, t_c(I) = 33 / 86 / 208 / 499, t_c(C) = 46 / 128 / 328 / 814; t_c(I)/N = 4,71 / 4,78 / 4,43 / 4,06 (Beobachtung an vier Größen, kein Beweis); t_c(C) ist größer als t_c(I) | `test_k9` |
| Metropolis, gemeinsames u, λ = 1 | in n = 4, 6, 8, 10 hat kein Paar (0 von 42 / 306 / 2162 / 15 006) einen Übergang auf die Diagonale; bei λ = 2 und 5 (n = 8) gibt es ihn; t_mix der Kette ist endlich | `test_k10` |
| Zwei Startzustände genügen | das Maximum von P(τ > t) über alle Paare ist das des Gipfelpaars: 24 Zellen (S, gerades n), 4 Zellen (S, ungerades n: n = 7, 9; λ = 1, 5), 2 Zellen (Metropolis, λ = 5), größter Unterschied 2e-15 (bewiesen nur mit Monotonie); ungerades n: t_c/t_mix = 2,27 / 2,51 / 2,21 / 2,61 | `test_k11` |
| Ungleichungen und Satz 5 | an 1379 Stützstellen (n = 4 bis 10, ℓ = n): d(t) ≤ P(τ > t), P(τ > t)/ℓ ≤ d̄(t) ≤ P(τ > t) (nach dem Quellenbericht) und d ≤ d̄ ≤ 2d: 0 Verletzungen; die Verwechslung von d und d̄ ist am Ring in 0 von 1379 Stützstellen sichtbar | `test_k12` |
| Falle d gegen d̄ (drei Zustände) | P(T > 1)/2 = 1/2 liegt über d(1) = 1/3 und ist gleich d̄(1) = 1/2; d̄(k) = 2^(−k) für k = 1 bis 10; E[τ] = 3 | `test_k13`, `test_handbeispiel.py` |
| Läufe gegen exakt (SplitMix64) | n = 8, λ = 5: 1500 Läufe, E[τ] im Band (Abstand unter 4 Standardfehlern) gegen 265,32, Kolmogorov-Abstand höchstens 2·1,36/√R; ebenso für S (n = 6, λ = 2; 3000 Läufe), C, I, M | `test_oracle_kopplung_ketten.py` |
| Ohne Matrix: Läufe-Tabelle (vorgerechnet) | λ = 1, n = 8 / 24 / 40: E[τ] 52,9 ± 0,7 / 237,6 ± 2,2 / 451,9 ± 4,9, q75 66 / 287 / 541; λ = 5: n = 8 / 24 / 40: 269,2 ± 4,4 / 1252,8 ± 14,3 / 2538,1 ± 37,6, q75 354 / 1564 / **3091**; λ = 20, n = 12 / 24: 3825,6 ± 66,3 / 10 085,2 ± 235,5, q75 4986 / 12 743; 19 Zeilen, 0 Läufe ohne Treffen | `test_k14` |
| q75 gegen exakte Mischzeit | in den 15 Zeilen mit bekannter Mischzeit (n = 8 bis 24) liegt q75 beim 2,02- bis 2,40-fachen von t_mix, n = 24, λ = 5: 1564 gegen 705 (2,22-fach); n ≥ 16: t_mix nur aus dem Start der geraden Plätze (Untergrenze); für n ≤ 12 ist dieser Start der schlechteste | `test_k15` |
| E[τ] gegen n ln n und t_rel | E[τ]/(n ln n) bei λ = 1 über n = 8 bis 40: 3,06 bis 3,18; bei λ = 5: 16,04 bis 17,20 (Beobachtung für n ≤ 40, kein Beweis); E[τ]/t_rel wächst mit n (λ = 1: 1,93 bei n = 8, 3,31 bei n = 40; λ = 5: 1,45 bis 2,73) | `test_k16` |
| Stichproben-TV | bei λ = 1 ist π gleichverteilt; eine Stichprobe von R Ziehungen hat TV ≥ 1 − R/N: n = 24, R = 10 000: mindestens 0,90; n = 40, R = 10⁵: mindestens 0,9996 (Rechnung, keine Messung) | `test_k17` |
| Wie viele Läufe? n = 10, λ = 20 (exakt: t_mix 1805, t_c 3967) | 5-%-/95-%-Band des aus R Läufen geschätzten 75-%-Quantils: R = 10: 2261 bis 6499; 20: 2521 bis 5370; 50: 3091 bis 5023; 200: 3485 bis 4447; 1000: 3750 bis 4183; Wahrscheinlichkeit unter t_mix: 1,02 % (R = 10), 0,12 % (R = 20), unter 0,01 % ab R = 50 | `test_k18` |

Messungen dieser README, die **nicht** im Test stehen: die Rechenzeiten (Abschnitt Methodik) und die Zahlen der ersten Rechnung im Abschnitt „Befunde und Korrekturen“.

**Negativbefunde (bleiben stehen):**
* **Eine Kopplung kann leer bleiben, obwohl die Kette schnell mischt:** Metropolis mit gemeinsamem u trifft sich bei λ = 1 nie, bei n = 8 mischt die Kette in 15 Schritten. „Kein Treffen“ sagt über die Mischzeit nichts.
* **Die Mittelwert-Schranke 4·E[τ] ist grob** (5,79- bis 7,07-fach), die Quantil-Schranke liegt beim 1,89- bis 2,28-fachen; die Quantil-Schranke ist die brauchbare.
* **E[τ] wächst bei λ = 1 etwa wie n ln n:** E[τ]/(n ln n) liegt über n = 8 bis 40 bei 3,06 bis 3,18; bei λ = 5 steigt das Verhältnis leicht (16,04 bis 17,20); E[τ]/t_rel wächst mit n (Beobachtung für n ≤ 40, kein Beweis).
* **Die erwartete Falle „zu wenige Läufe“ ist in der gemessenen Zelle (n = 10, λ = 20) nicht zu sehen:** die aus 10 Läufen geschätzte Schranke liegt nur in 1,02 % unter der Mischzeit, ab 50 Läufen in unter 0,01 %. Ob das bei stärkerer Mehrgipfligkeit (λ = 100) anders ist, ist nicht gemessen.
* **Die Falle d gegen d̄ ist am Ring unsichtbar** (0 von 1379 Stützstellen); sie zeigt sich nur bei kleinem ℓ, am Beispiel mit drei Zuständen.

## Befunde und Korrekturen gegenüber der ersten Rechnung

* **Das Band der Schätzung ist hier exakt gerechnet statt simuliert:** die erste Rechnung gab aus 4000 simulierten Wiederholungen bei R = 10 das Band 2201 bis 6115 und 1,2 % unter der Mischzeit, bei R = 20 0,2 %; die Ordnungsstatistik gibt 2261 bis 6499 und 1,02 % bzw. 0,12 %. Die Zahlen der README sind die exakten.
  Die Wiederholungen in der App werden aus der exakten Verteilung gezogen, nicht aus neuen Läufen.
* **Die Stützstellen reichen bis zum Zehnfachen von t̄_mix statt bis zum Vierfachen:** die Schranke für ε = 0,001 liegt bei n = 10, λ = 20 weit über 4·t̄_mix. Darum sind es 1379 Stützstellen statt 1285; es bleibt bei 0 Verletzungen.
* **Läufe-Tabelle mit anderem Zufallsstrom:** aus SplitMix64 statt aus numpy. q75 liegt in 15 Zeilen beim 2,02- bis 2,40-fachen von t_mix (die erste Rechnung: 12 Zeilen, 2,04 bis 2,33; die Zeile n = 16, λ = 20 kam dazu und gibt mit 600 Läufen 2,40). Einzelne Zahlen weichen im Rahmen ihrer Bänder ab,
  etwa E[τ] bei n = 24, λ = 20: 10 085,2 ± 235,5 gegen 10 543,6 ± 275,6 (600 Läufe). Das Band ist der Befund, nicht die letzte Stelle.
* **Metropolis bei λ = 1 ist schärfer als „nicht monoton“:** die Standardkopplung trifft sich aus keinem Paar (nicht nur nicht monoton); bei λ = 5 gelingt sie (n = 8: 332, 2,61-fach).
* **λ ist ein Gitter** (0,5; 1; 2; 5; 10; 20), kein freier Regler: die Zahlen der README stehen an diesen Werten; die Läufe in Experiment 1 rechnen live, die exakte Verteilung nur für n ≤ 8 (n = 10 für Kopplung S und das Gipfelpaar vorgerechnet).
* **Die Gleichheit d̄ = 2d** (n = 4, λ = 20, an zwei Stützstellen, in 12 Stellen gespeichert) liegt auf der Grenze der Ungleichung d̄ ≤ 2d; die Prüfung hat deshalb eine Toleranz von 1e-9.

## Ehrliche Grenzen

* **Ausgedachtes Modell:** keine reale Datenquelle; keine Messung in der Literatur, dass MCMC im Betrieb bessere Entscheidungen liefert, ist hier belegt oder behauptet.
* **Vier Kopplungen, keine optimierte.** Ob es eine Kopplung mit P(τ > t) = d̄(t) gibt, ist nicht gemessen; die Aussage „die Schranke ist scharf, wenn man richtig koppelt“ steht hier nicht. Am nächsten an d̄ liegt die Kopplung S bei n = 4, λ = 20 (t_c/t̄_mix = 1,00, der Faktor 2 ist dort die Zähleinheit).
* **Nur zwei Modelle** (Hard-Core-Ring und Hyperwürfel); Ising-Kreis und Gaußmischung sind nicht gemessen; ähnliche Faktoren heißen nicht mehr.
* **Gerades n für die Monotonie;** bei ungeradem n sind nur vier Zellen gemessen, nur für Kopplung S. „Zwei Startpunkte reichen“ ist nur mit Monotonie bewiesen, in den gemessenen Zellen gilt es auch ohne.
* **τ ist eine Zeit, keine Probe.** Hört man beim ersten Treffen auf und nimmt den Zustand als Probe, ist sie verzerrt; das gehört in `kopplung-vergangenheit-demo`. Kopplung verbessert das Mischen der Einzelketten nicht (Jacob/O'Leary/Atchadé 2020).
* **Propp/Wilson, Satz 5** ist nur nach dem Quellenbericht geführt (Wortlaut des Originals nicht erneut geöffnet); gemessen ist er im Modell. Die Formel t_rel = n/g(λ) (nur Vergleichsspalte, gerades n) stammt aus der Nachbar-Demo `mischzeit-exakt-demo`, nicht aus einer Quelle.
* **Zahlen aus Läufen** gelten für den Zufalls-Seed der Zeile; die Bänder sind der Befund. Plattform der Messung: Windows, Python 3.14; Ganzzahlen und Brüche sind plattformfest, der SplitMix64 ebenso.
* **Nicht gelesen und nicht zitiert:** Norris, Handbook of Markov Chain Monte Carlo (Buch), Robert/Casella, Sinclair/Jerrum 1989, Besag 1974; Gelman/Rubin 1992 nur zwei Seiten.

## Verifikation

Die Suite hat **240 Tests** (`pytest --collect-only -q tests`); der Test `test_readme_nennt_die_testzahl_der_suite` hält die Zahl aktuell.

- **Handbeispiel:** `tests/test_handbeispiel.py` rechnet den Kleinstfall (n = 4, λ = 1) in exakten Brüchen mit eigenem Code und vergleicht die numpy-Rechnung der Demo damit.
- **Orakel:** `tests/test_oracle_kopplung_ketten.py`: Zustandsraum gegen Aufzählen aller 2^n Masken; Heat-bath- und Metropolis-Matrix gegen eine zweite Konstruktion (Metropolis als I + Q/(n·max(1, λ))); Paarkette gegen Vorwärtsrechnung aus den Zufallsquellen für alle vier Kopplungen;
  Ausspielen aller Folgen (n = 6, t ≤ 4); Randketten der Kopplungen S, C, M gleich der Übergangsmatrix; d und d̄ aus Spektralzerlegung gegen Matrixpotenz; E[τ] gegen die Reihe; Sammlerformel gegen Zählkette und Paarkette; Mischzeit des Hyperwürfels gegen alle 2^m Starts;
  simulierte Treffzeiten gegen die exakte Kurve; Ordnungsstatistik gegen Aufzählen aller Ausgänge.
- **Eigenschaften:** `tests/test_eigenschaften_kopplung_ketten.py` (Kurve fällt, τ ≥ n bei gemeinsamem Platz, Ungleichungskette über das Raster, Maximum über alle Paare, Drehung und Spiegelung, Schranke für alle vier Kopplungen); **Literatur:** `tests/test_literatur_kopplung_ketten.py` (nur Gelesenes, Quelle im Docstring).
- **Vorgerechnete Dateien:** `tests/test_reference.py` rechnet n ≤ 8 vollständig neu und n = 10 an Stichproben; vollständig mit `python tools/kpl_make_curves.py --check --lang` und `python tools/kpl_make_scaling.py --check`.
- **Verdikte und Randfälle:** `tests/test_verdicts.py` (jede Zahl im Text), `tests/test_edge_cases.py` (n < 4, ungerades n, falsche Eingaben, Obergrenzen, Metropolis bei λ = 1); **Zahlen der README:** `tests/test_claims.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an beiden Rändern, jede Kopplung), Layout (Schnellstart oben in der Seitenleiste, fester Kopf, aktives Beispiel hervorgehoben; jede Änderung eines Reglers hebt die Markierung auf), Footer und Laufzeit-Importe.
  Ein Test prüft, dass keine Begriffe aus der Planung in Texten, README und Tests stehen.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore`, `.streamlit/config.toml` | Caches, Secrets; Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: vier Experimente, Grenzen, Formeln |
| `kpl_chain.py` | Ring: Zustände, Übergangsmatrizen, Aktualisierungsfunktionen, die vier Kopplungen als Paarkerne, Einzelschritte |
| `kpl_pairs.py` | Paarkette: P(τ > t), E[τ], erreichbare Paare, „trifft nie“ |
| `kpl_exact.py` | d, d̄, Mischzeiten, Ordnung, Quantile, Band der Schätzung, Hyperwürfel, drei Zustände |
| `kpl_sim.py` | gekoppelte Läufe (SplitMix64), Schrittbudget, Eingabe von Belegungen |
| `kpl_rng.py` | SplitMix64 (geprüfte Vorlage, unverändert) |
| `kpl_reference.py`, `kpl_curves.json`, `kpl_scaling.json` | vorgerechnete Daten und ihr Zugriff |
| `kpl_evaluation.py` | Auswertung der Experimente und Verdikte (reine Funktionen) |
| `kpl_constants.py`, `kpl_presets.py`, `kpl_layout.py`, `kpl_format.py`, `kpl_texts.py`, `kpl_visualization.py` | Regler-Grenzen und Schnellstart, Permalink, fester Kopf, Zahlenformat, Texte, Abbildungen |
| `tools/kpl_make_curves.py`, `tools/kpl_make_scaling.py` | erzeugen und prüfen die vorgerechneten Dateien |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit-Bibliotheken (ungepinnt), dazu pytest |
| `tests/` | Handbeispiel, Orakel, Eigenschaften, Literatur, Vorgerechnetes, Verdikte, Randfälle, Zahlen der README, Oberfläche, Layout, Footer, Laufzeit-Importe, Begriffe |

## Bewusst nicht umgesetzt

* **Zeitversetzte Kopplung** (Schranken für d(t) selbst ohne Matrix, Biswas/Jacob/Vanetti 2019): der Wortlaut der Formel ist nicht gelesen.
* **λ = 100** (zwei scharfe Gipfel): die Läufe wären zu lang; dort ist auch die Falle „zu wenige Läufe bei zwei Gipfeln“ offen.
* **Eine optimierte Kopplung** und **Metropolis bei λ = 2 und 10** oder mit einem anderen Zufallsschema für u; ein **Ising-Kreis** und die **Gaußmischung** als zweites Vehikel.
* **Freies λ** und exakte Verteilungen für n > 8 live: das Gitter der vorgerechneten Werte genügt für jede Zahl der README.
* **Die Varianz einer Differenz bei gemeinsamen Zufallszahlen** (die Monte-Carlo-Bedeutung von „Kopplung“).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [MCMC: von der Kette zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-mcmc.html).
