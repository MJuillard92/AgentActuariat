"""Reproduit le bug user : après un premier rapport descriptif, demander
"construit maintenant les taux bruts et le rapport associé" ne lance
aucun calcul."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

for line in (_PROJECT_ROOT / ".env").read_text().splitlines():
    if line.startswith("OPENAI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')


def _print_turn_summary(turn_num: int, prompt: str, events: list) -> None:
    print(f"\n{'=' * 78}")
    print(f"TOUR {turn_num} — {prompt!r}")
    print('=' * 78)
    for i, ev in enumerate(events):
        t = ev.get("type")
        if t == "tool_call":
            print(f"  {i:3d} TOOL {ev.get('tool')}.{ev.get('function_name')}")
        elif t == "message":
            content = (ev.get("content") or "")[:160]
            if content:
                print(f"  {i:3d} MSG {content}")
        elif t == "agent_switch":
            print(f"  {i:3d} → {ev.get('agent')}")
        elif t == "error":
            print(f"  {i:3d} ERROR {(ev.get('message') or '')[:200]}")
        elif t == "report_ready":
            print(f"  {i:3d} REPORT {ev.get('output_path')} ({ev.get('nb_sections')} sec)")


def main() -> int:
    from session.memory_manager import MemoryManager
    from agents.mortality.agents.graph import stream_agent

    df = pd.read_csv(_PROJECT_ROOT / "Portefeuille" / "portefeuille_test_1000.csv")
    session = f"sim_followup_raw_{int(time.time())}"

    mm = MemoryManager(session); mm.load()
    mm.register_dataset(df, csv_filename="portefeuille_test_1000.csv")
    mm.state.column_mapping = {
        "date_naissance": "date_naissance", "date_entree": "date_entree",
        "date_sortie":    "date_sortie",    "cause_sortie": "cause_sortie",
        "sexe":           "sexe",
    }
    mm.state.column_mapping_confirmed = True
    mm.state.disambiguation_done = True
    mm.save()

    history = []

    # Tour 1 : descriptif unisex
    history.append({"role": "user", "content": "construit un rapport d'analyse descriptive unisex"})
    events = []
    for ev in stream_agent(history=history, data_store=mm.to_data_store(), thread_id=session):
        events.append(ev)
        if len(events) > 100:
            break
    _print_turn_summary(1, history[-1]["content"], events)
    # Récupérer la dernière réponse assistant
    last_msg = next((e.get("content") for e in reversed(events)
                     if e.get("type") == "message" and e.get("content")), "")
    history.append({"role": "assistant", "content": last_msg})

    # Tour 2 : taux bruts
    history.append({"role": "user", "content": "construit maintenant les taux bruts et le rapport associé"})
    events = []
    for ev in stream_agent(history=history, thread_id=session):
        events.append(ev)
        if len(events) > 100:
            break
    _print_turn_summary(2, history[-1]["content"], events)

    # Inspect final state
    mm2 = MemoryManager(session); mm2.load()
    final_ds = mm2.to_data_store()
    print("\n" + "=" * 78)
    print("ÉTAT FINAL data_store :")
    print('=' * 78)
    keys = ["_write", "_kind", "report_mode", "_master_builder_cycles",
            "qx_table", "exposure_table", "smoothed_table",
            "total_exposure", "total_deaths", "exclusion_report",
            "section_outputs"]
    for k in keys:
        v = final_ds.get(k)
        if v is None:
            display = "MISSING"
        elif isinstance(v, dict):
            display = f"dict({len(v)})"
        elif isinstance(v, list):
            display = f"list({len(v)})"
        else:
            display = str(v)[:80]
        print(f"  {k:30} = {display}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
