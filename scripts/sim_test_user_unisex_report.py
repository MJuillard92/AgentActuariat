"""Reproduit le bug user : 'construit un rapport d'analyse descriptive unisex'
ne génère qu'une seule section (data_preprocessing). Préambule et
data_analysis_unisex sont skippés."""
from __future__ import annotations

import json
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


def main() -> int:
    from session.memory_manager import MemoryManager
    from session.session_state import StudyPlan
    from agents.mortality.agents.graph import stream_agent

    df = pd.read_csv(_PROJECT_ROOT / "Portefeuille" / "portefeuille_test_1000.csv")
    session = f"sim_user_unisex_{int(time.time())}"

    mm = MemoryManager(session); mm.load()
    mm.register_dataset(df, csv_filename="portefeuille_test_1000.csv")
    mm.state.column_mapping = {
        "date_naissance": "date_naissance", "date_entree": "date_entree",
        "date_sortie":    "date_sortie",    "cause_sortie": "cause_sortie",
        "sexe":           "sexe",
    }
    mm.state.column_mapping_confirmed = True
    mm.state.disambiguation_done = True
    # NOTE : on simule un upload CSV sans paramètres d'étude (cas réel
    # canvas_app : l'utilisateur upload le CSV puis tape directement la
    # demande sans renseigner explicitement observation_start/end_date).
    # mm.state.study_plan est laissé vide.
    mm.save()

    history = [{"role": "user", "content": "construit un rapport d'analyse descriptive unisex"}]
    events = []
    for ev in stream_agent(history=history, data_store=mm.to_data_store(), thread_id=session):
        events.append(ev)
        if len(events) > 80:
            break

    # Inspect final data_store
    mm2 = MemoryManager(session); mm2.load()
    final_ds = mm2.to_data_store()

    # Count per-key presence (avec valeurs effectives)
    keys_to_check = [
        "study_objective", "start_year", "end_year", "num_observation_years",
        "total_exposure", "total_deaths", "total_records",
        "segmentations", "serie", "ages",
        "exclusion_report",
        "section_outputs",
    ]
    print("DATA STORE FINAL KEYS:")
    for k in keys_to_check:
        v = final_ds.get(k)
        if v is None:
            display = "MISSING"
        elif isinstance(v, dict):
            display = f"dict({len(v)} keys: {list(v.keys())[:5]})"
        elif isinstance(v, list):
            display = f"list({len(v)})"
        else:
            display = str(v)[:60]
        print(f"  {k:25} = {display}")

    # Print events
    print("\nEVENTS:")
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
        elif t == "report_ready":
            print(f"  {i:3d} REPORT {ev}")
        elif t == "error":
            print(f"  {i:3d} ERROR {(ev.get('message') or '')[:200]}")
    print()

    # Check section_outputs
    so = final_ds.get("section_outputs") or {}
    print("SECTION_OUTPUTS:")
    for sid, sec in so.items():
        print(f"  {sid:30}  status={sec.get('status'):10} text_len={len(sec.get('text','') or '')}  "
              f"tables={len(sec.get('tables') or [])} graphs={len(sec.get('graphs') or [])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
