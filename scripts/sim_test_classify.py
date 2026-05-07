"""Niveau 1 — Test de classification 3 axes (kind, write, report_mode).

Exécute _classify_intent sur 8 phrases représentatives et affiche les axes
attendus vs obtenus. Sortie compacte pour validation manuelle.

Usage :
    python scripts/sim_test_classify.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

from agents.mortality.agents.master_node import _classify_intent  # noqa: E402


# (phrase, kind attendu, write attendu, report_mode attendu)
SCENARIOS: list[tuple[str, str, str, str]] = [
    ("construis-moi une table de mortalité",
     "task", "ask", "full_report"),

    ("fais-moi le rapport avec les taux bruts",
     "task", "yes", "raw_rates"),

    ("fais-moi une analyse descriptive",
     "task", "ask", "description"),

    ("rédige-moi un rapport descriptif",
     "task", "yes", "description"),

    ("calcule les taux lissés sans rapport",
     "task", "no", "full_report"),

    ("pas de PDF, juste les taux bruts",
     "task", "no", "raw_rates"),

    ("c'est quoi le lissage Whittaker ?",
     "question", "*", "*"),

    ("je veux un rapport complet",
     "task", "yes", "full_report"),
]


def _ok(actual: str, expected: str) -> str:
    """Compare avec un wildcard '*' qui ignore le check."""
    if expected == "*":
        return f"({actual})"
    return "✓" if actual == expected else f"✗ → {actual!r} (attendu : {expected!r})"


def main() -> int:
    print("=" * 80)
    print(f"Niveau 1 — Classification 3 axes ({len(SCENARIOS)} scénarios)")
    print("=" * 80)

    failures = 0
    for phrase, exp_kind, exp_write, exp_report in SCENARIOS:
        result = _classify_intent(phrase, data_store={}, dataset_ref=None)
        kind        = result.get("kind", "?")
        write       = result.get("write", "?")
        report_mode = result.get("report_mode", "?")
        reply       = (result.get("reply") or "").strip()

        kind_check   = _ok(kind, exp_kind)
        write_check  = _ok(write, exp_write)
        report_check = _ok(report_mode, exp_report)

        any_failure = "✗" in (kind_check + write_check + report_check)
        if any_failure:
            failures += 1

        print(f"\n  [{('FAIL' if any_failure else 'OK ')}] « {phrase} »")
        print(f"     kind        : {kind_check}")
        print(f"     write       : {write_check}")
        print(f"     report_mode : {report_check}")
        if reply:
            print(f"     reply       : {reply[:120]}")

    print("\n" + "=" * 80)
    print(f"Total : {len(SCENARIOS) - failures}/{len(SCENARIOS)} OK | {failures} échec(s)")
    print("=" * 80)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
