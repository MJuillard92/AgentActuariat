"""Test end-to-end Master sur la phrase qui pose problème en UX."""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")


PHRASE = "construit les taux bruts et le rapport associé"


def main() -> int:
    print(f"\n  Phrase utilisateur : {PHRASE!r}\n")

    from agents.mortality.agents import master_node as mn

    state = {
        "messages":    [HumanMessage(content=PHRASE)],
        "data_store":  {
            "_disambiguation_done":   True,
            "study_plan":             {"gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
    }
    out = mn.master_node(state)

    print(f"  → active_agent      : {out.get('active_agent')}")
    ds = out.get("data_store") or {}
    print(f"  → data_store flags  :")
    for k in ("_kind", "_write", "report_mode", "_write_question_asked",
              "_master_builder_cycles", "_intent"):
        if k in ds:
            print(f"      {k:30s} = {ds[k]!r}")

    print(f"\n  → messages produits ({len(out.get('messages') or [])}) :")
    for m in (out.get("messages") or []):
        cls = type(m).__name__
        content = (getattr(m, "content", "") or "")
        print(f"      {cls} : {content[:200]}")

    print(f"\n  → events :")
    for ev in (out.get("events") or []):
        if isinstance(ev, dict):
            t = ev.get("type")
            c = ev.get("content") or ev.get("agent") or ""
            print(f"      {t}: {str(c)[:120]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
