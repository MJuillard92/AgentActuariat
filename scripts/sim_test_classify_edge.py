"""Test ciblé : phrases edge-case relevées par l'utilisateur.

Le user observe que des phrases comme "construit les taux bruts et le rapport
associé" déclenchent à tort la question de désambiguation PDF.
On vérifie ce que le mini-LLM retourne sur ces phrases-pièges.
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


PHRASES = [
    # (phrase, expected_kind, expected_write, expected_report_mode)
    ("construit les taux bruts et le rapport associé", "task", "yes", "raw_rates"),
    ("construis les taux bruts et le rapport associé", "task", "yes", "raw_rates"),
    ("calcule les taux bruts et fais le rapport",      "task", "yes", "raw_rates"),
    ("fais une analyse descriptive et le rapport",     "task", "yes", "description"),
    ("je veux un rapport sur les taux bruts",          "task", "yes", "raw_rates"),
    ("construit la table de mortalité avec rapport",   "task", "yes", "full_report"),
    ("rapport descriptif please",                       "task", "yes", "description"),
]


def _ok(actual, expected):
    if expected == "*":
        return f"({actual})"
    return "✓" if actual == expected else f"✗ → got {actual!r}, expected {expected!r}"


def main() -> int:
    print("=" * 78)
    print(f"  Edge-case classify_intent : {len(PHRASES)} phrases ambiguës")
    print("=" * 78)
    fail_count = 0
    for phrase, exp_kind, exp_write, exp_mode in PHRASES:
        result = _classify_intent(phrase, data_store={}, dataset_ref=None)
        kind = result.get("kind", "?")
        write = result.get("write", "?")
        mode = result.get("report_mode", "?")
        reply = (result.get("reply") or "").strip()

        kk = _ok(kind, exp_kind)
        wk = _ok(write, exp_write)
        mk = _ok(mode, exp_mode)
        any_fail = "✗" in (kk + wk + mk)
        if any_fail:
            fail_count += 1

        print(f"\n  [{('FAIL' if any_fail else 'OK ')}] « {phrase} »")
        print(f"     kind        : {kk}")
        print(f"     write       : {wk}")
        print(f"     report_mode : {mk}")
        if reply:
            print(f"     reply       : {reply[:120]}")

    print("\n" + "=" * 78)
    print(f"  Total : {len(PHRASES) - fail_count}/{len(PHRASES)} OK | {fail_count} échec(s)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
