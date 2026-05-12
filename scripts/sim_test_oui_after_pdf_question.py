"""Reproduit le bug user : après "voulez-vous un PDF ?", la réponse "oui"
est classifiée à tort comme `kind=question`."""
from __future__ import annotations

import os, sys, time
from pathlib import Path
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

for line in (_PROJECT_ROOT / ".env").read_text().splitlines():
    if line.startswith("OPENAI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')


def main() -> int:
    from session.memory_manager import MemoryManager
    from agents.mortality.agents.graph import stream_agent

    df = pd.read_csv(_PROJECT_ROOT / "Portefeuille" / "portefeuille_test_1000.csv")
    session = f"sim_oui_{int(time.time())}"
    mm = MemoryManager(session); mm.load()
    mm.register_dataset(df, "portefeuille_test_1000.csv")
    mm.state.column_mapping = {
        "date_naissance": "date_naissance", "date_entree": "date_entree",
        "date_sortie":    "date_sortie",    "cause_sortie": "cause_sortie",
        "sexe":           "sexe",
    }
    mm.state.column_mapping_confirmed = True
    mm.state.disambiguation_done = True
    mm.save()

    history = []

    # Tour 1 : "construit un rapport d'analyse descriptive unisex" — mais SANS le mot "rapport"
    # Pour forcer le write=ask, on dit juste "analyse descriptive unisex"
    history.append({"role": "user", "content": "construit une analyse descriptive unisex"})
    events1 = list(stream_agent(history=history,
                                data_store=mm.to_data_store(),
                                thread_id=session))
    last_ai = next((e.get("content") for e in reversed(events1)
                    if e.get("type") == "message"), "")
    history.append({"role": "assistant", "content": last_ai})

    print(f"\n══ TOUR 1 — {history[0]['content']!r}")
    for ev in events1:
        t = ev.get("type")
        if t == "message":
            c = (ev.get("content") or "")[:140]
            if c: print(f"  MSG {c}")
        elif t == "agent_switch":
            print(f"  → {ev.get('agent')}")
        elif t == "tool_call":
            print(f"  TOOL {ev.get('tool')}.{ev.get('function_name')}")

    # Tour 2 : "oui"
    history.append({"role": "user", "content": "oui"})
    events2 = list(stream_agent(history=history, thread_id=session))

    print(f"\n══ TOUR 2 — 'oui'")
    has_tool = False
    has_pdf = False
    for ev in events2:
        t = ev.get("type")
        if t == "message":
            c = (ev.get("content") or "")[:140]
            if c: print(f"  MSG {c}")
        elif t == "agent_switch":
            print(f"  → {ev.get('agent')}")
        elif t == "tool_call":
            has_tool = True
            print(f"  TOOL {ev.get('tool')}.{ev.get('function_name')}")
        elif t == "report_ready":
            has_pdf = True
            print(f"  REPORT {ev.get('output_path')} ({ev.get('nb_sections')} sec)")
        elif t == "error":
            print(f"  ERROR {(ev.get('message') or '')[:200]}")

    print(f"\n→ Tools called in tour 2 : {has_tool}")
    print(f"→ PDF generated in tour 2 : {has_pdf}")
    return 0 if has_tool and has_pdf else 1


if __name__ == "__main__":
    sys.exit(main())
