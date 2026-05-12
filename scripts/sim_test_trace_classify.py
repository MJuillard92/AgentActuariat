"""Trace pas-à-pas le retour du Master pour 'construit un rapport
d'analyse descriptive unisex' — montre la classification LLM, la décision
de routage, et tous les events émis."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

for line in (_PROJECT_ROOT / ".env").read_text().splitlines():
    if line.startswith("OPENAI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')


def main() -> int:
    from session.memory_manager import MemoryManager
    from agents.mortality.agents import master_node as mn
    from agents.mortality.agents.graph import stream_agent

    # ── Capture le retour brut de _classify_intent ──
    orig_classify = mn._classify_intent
    captured_classify: dict = {}

    def _capturing(last_human, data_store, dataset_ref):
        result = orig_classify(last_human, data_store, dataset_ref)
        captured_classify[last_human] = result
        return result

    mn._classify_intent = _capturing

    df = pd.read_csv(_PROJECT_ROOT / "Portefeuille" / "portefeuille_test_1000.csv")
    session = f"sim_trace_{int(time.time())}"
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

    user_msg = "construit un rapport d'analyse descriptive unisex"
    history = [{"role": "user", "content": user_msg}]

    events = list(stream_agent(history=history,
                               data_store=mm.to_data_store(),
                               thread_id=session))

    print(f"\n{'=' * 78}")
    print(f"USER : {user_msg!r}")
    print('=' * 78)

    print("\n┌── CLASSIFY_INTENT — retour brut LLM ──")
    cls = captured_classify.get(user_msg, {})
    for k, v in cls.items():
        print(f"│  {k:15} = {v!r}")
    print("└──────────────────────────────────────")

    print("\n┌── EVENTS émis par le graphe ──")
    for i, ev in enumerate(events):
        t = ev.get("type")
        if t == "agent_switch":
            print(f"│  {i:3d} switch  → {ev.get('agent')}")
        elif t == "message":
            content = (ev.get("content") or "")[:140]
            print(f"│  {i:3d} message  {content}")
        elif t == "tool_call":
            print(f"│  {i:3d} tool     {ev.get('tool')}.{ev.get('function_name')}")
        elif t == "report_ready":
            print(f"│  {i:3d} report   {ev.get('output_path')}")
        elif t == "done":
            print(f"│  {i:3d} done")
    print("└──────────────────────────────────────")

    # Final data_store snapshot
    mm2 = MemoryManager(session); mm2.load()
    ds = mm2.to_data_store()
    print("\n┌── DATA_STORE final (extraits) ──")
    for k in ("_kind", "_write", "report_mode",
              "_write_question_asked", "_master_builder_cycles"):
        print(f"│  {k:30} = {ds.get(k)!r}")
    sp = ds.get("study_plan") or {}
    print(f"│  study_plan.gender_segmentation = {sp.get('gender_segmentation')!r}")
    print("└──────────────────────────────────────")
    return 0


if __name__ == "__main__":
    sys.exit(main())
