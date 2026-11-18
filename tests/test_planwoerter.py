"""Begriffe aus der eigenen Planung stehen nicht in App-Texten, README, Docstrings und Tests (DEMO-PLAYBOOK §11a Punkt 16). Die Zeile „Mehr zur Reihe“ im Portfolio-Footer erzeugt set_demo_footer.py und ist kein Planungstext."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERBOTEN = re.compile(r"\b(Stück|Stücke|Stücks|Linie|Linien|Vormessung|Phase 0|Kantenabgleich|mc_v\w*|V1|V2|V3)\b|\bPlan\b|\bPläne\b|\bPlan-")


def _dateien():
    yield from sorted(ROOT.glob("*.py"))
    yield from sorted((ROOT / "tests").glob("*.py"))
    yield from sorted((ROOT / "tools").glob("*.py"))
    yield ROOT / "README.md"


def test_keine_planungswoerter():
    treffer = []
    for p in _dateien():
        if p.name == "test_planwoerter.py":
            continue
        for i, zeile in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "Mehr zur Reihe" in zeile or "Mehr zu den Demos" in zeile:
                continue
            if VERBOTEN.search(zeile):
                treffer.append(f"{p.name}:{i}: {zeile.strip()[:100]}")
    assert not treffer, "\n".join(treffer)


def test_kein_traegt_kein_wort_saatwert_kein_wort_waermebad():
    treffer = []
    for p in _dateien():
        if p.name == "test_planwoerter.py":
            continue
        text = p.read_text(encoding="utf-8")
        for wort in ("trägt", "tragen", "Saatwert", "Wärmebad", "läuft lokal"):
            if wort in text:
                treffer.append(f"{p.name}: {wort}")
    assert not treffer, treffer
