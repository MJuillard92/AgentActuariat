"""Test ciblé : extraction de gender_segmentation depuis le message user.

Reproduit le cas reporté : "construit un rapport d'analyse descriptive unisex"
ne devrait PAS reposer la question du sexe."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# OPENAI key
for line in (_PROJECT_ROOT / ".env").read_text().splitlines():
    if line.startswith("OPENAI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')


def _bootstrap(session_id: str, df: pd.DataFrame, set_gender: str | None = None) -> dict:
    """Comme l'UI : column_mapping confirmé, gender_segmentation **NON** posé."""
    from session.memory_manager import MemoryManager
    from session.session_state import StudyPlan
    mm = MemoryManager(session_id)
    mm.load()
    mm.register_dataset(df, csv_filename="portefeuille_test_1000.csv")
    mm.state.column_mapping = {
        "date_naissance": "date_naissance",
        "date_entree":    "date_entree",
        "date_sortie":    "date_sortie",
        "cause_sortie":   "cause_sortie",
        "sexe":           "sexe",
    }
    mm.state.column_mapping_confirmed = True
    mm.state.disambiguation_done = True
    sp_dict = mm.state.study_plan.model_dump()
    sp_dict["observation_start_date"] = "2018-01-01"
    sp_dict["observation_end_date"] = "2022-12-31"
    sp_dict["baseline_regulatory_table"] = "TH-TF-00-02"
    if set_gender:
        sp_dict["gender_segmentation"] = set_gender
    mm.state.study_plan = StudyPlan(**sp_dict)
    mm.save()
    return mm.to_data_store()


def _run_one(prompt: str, expect_gender: str | None) -> dict:
    """Run un seul tour. Retourne (final_data_store, events, asked_gender_q)."""
    from agents.mortality.agents.graph import stream_agent

    df = pd.read_csv(_PROJECT_ROOT / "Portefeuille" / "portefeuille_test_1000.csv")
    session = f"test_gender_{int(time.time()*1000)}"
    ds = _bootstrap(session, df, set_gender=None)

    history = [{"role": "user", "content": prompt}]
    events = []
    asked_gender_q = False
    for ev in stream_agent(history=history, data_store=ds, thread_id=session):
        events.append(ev)
        if ev.get("type") == "message" and "table agrégée" in (ev.get("content") or "").lower():
            asked_gender_q = True
        if len(events) > 60:
            break

    # Inspecter le data_store final
    from session.memory_manager import MemoryManager
    mm = MemoryManager(session); mm.load()
    final_ds = mm.to_data_store()
    final_gender = (final_ds.get("study_plan") or {}).get("gender_segmentation")

    return {
        "prompt":           prompt,
        "expected_gender":  expect_gender,
        "final_gender":     final_gender,
        "asked_gender_q":   asked_gender_q,
        "n_events":         len(events),
        "ok":               (final_gender == expect_gender) and (not asked_gender_q),
    }


def main() -> int:
    cases = [
        ("construit un rapport d'analyse descriptive unisex",  "unisex"),
        ("Fais l'analyse descriptive en mode unisex",          "unisex"),
        ("Fais le rapport agrégé sans distinction de sexe",    "unisex"),
        ("Construis le rapport descriptif H/F",                "by_sex"),
        ("Fais l'analyse par sexe",                            "by_sex"),
        ("Fais une étude hommes et femmes séparées",           "by_sex"),
    ]
    n_ok = 0
    for prompt, expect in cases:
        r = _run_one(prompt, expect)
        ok = "✓" if r["ok"] else "✗"
        print(f"{ok}  prompt={prompt!r:65}  expected={expect:8}  got={str(r['final_gender']):8}  asked_q={r['asked_gender_q']}")
        if r["ok"]:
            n_ok += 1
    print(f"\n{n_ok}/{len(cases)} cases passed")
    return 0 if n_ok == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
