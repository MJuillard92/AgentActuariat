"""Battery 10 scénarios post-redaction RAG.

Couvre les 3 modes (description, raw_rates, full_report) × {avec/sans rapport}
× {unisex, by_sex}. Compare la sortie contre le rapport de référence
AF8796-TD3 sur des métriques structurelles (tableaux, graphiques, longueur).
"""
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

import logging
logging.basicConfig(level=logging.WARNING,
                     format="%(asctime)s %(name)s %(levelname)s %(message)s")

RUNS = _PROJECT_ROOT / "tmp" / "battery_redaction"
RUNS.mkdir(parents=True, exist_ok=True)


def _bootstrap(sid: str) -> dict:
    """Bootstrap minimal — column_mapping confirmé + dates d'observation."""
    from session.memory_manager import MemoryManager
    from session.session_state import StudyPlan

    df = pd.read_csv(_PROJECT_ROOT / "Portefeuille" / "portefeuille_test_1000.csv")
    mm = MemoryManager(sid); mm.load()
    mm.register_dataset(df, csv_filename="portefeuille_test_1000.csv")
    mm.state.column_mapping = {
        "date_naissance": "date_naissance", "date_entree": "date_entree",
        "date_sortie":    "date_sortie",    "cause_sortie": "cause_sortie",
        "sexe":           "sexe",
    }
    mm.state.column_mapping_confirmed = True
    mm.state.disambiguation_done = True
    sp = mm.state.study_plan.model_dump()
    sp["observation_start_date"] = "2018-01-01"
    sp["observation_end_date"] = "2022-12-31"
    mm.state.study_plan = StudyPlan(**sp)
    mm.save()
    return mm.to_data_store()


def _run_turn(sid: str, msg: str, data_store: dict | None = None) -> dict:
    """Lance un tour. Retourne dict structuré."""
    from session.memory_manager import MemoryManager
    from agents.mortality.agents import master_node as mn
    from agents.mortality.agents.graph import stream_agent

    # Capture classify
    orig = mn._classify_intent
    captured = {}
    def _spy(last_human, ds, dr):
        r = orig(last_human, ds, dr)
        captured[last_human] = dict(r)
        return r
    mn._classify_intent = _spy

    from agents.mortality.agents.graph import _checkpointer
    # Pour s'assurer que les events sont propres à chaque scénario, on
    # repart d'un thread vierge (la state persiste sur disque uniquement
    # via SessionState, le MemorySaver est ré-utilisé).

    history = []
    history.append({"role": "user", "content": msg})
    events = []
    for ev in stream_agent(history=history,
                            data_store=data_store,
                            thread_id=sid):
        events.append(ev)

    mn._classify_intent = orig

    mm = MemoryManager(sid); mm.load()
    return {
        "msg":      msg,
        "classify": captured.get(msg, {}),
        "events":   events,
        "ds":       mm.to_data_store(),
    }


def _analyze_pdf(pdf_path: str) -> dict:
    """Extrait les métriques structurelles depuis un PDF."""
    try:
        from pypdf import PdfReader
        r = PdfReader(pdf_path)
        full_text = "\n".join(p.extract_text() for p in r.pages)
        # Compte heuristique
        n_tables = full_text.count("Année") + full_text.count("Sexe")  # mots tabulaires
        # Repère sections
        section_markers = {
            "preamble":             "Préambule",
            "data_preprocessing":   "Données et prétraitement",
            "data_analysis_unisex": "Analyse descriptive du portefeuille",
            "data_analysis_by_sex": "Analyse descriptive H/F",
            "table_construction":   "Construction de la table",
        }
        sections_present = {sid: lbl in full_text for sid, lbl in section_markers.items()}
        return {
            "pages":       len(r.pages),
            "n_chars":     len(full_text),
            "n_words":     len(full_text.split()),
            "sections":    sections_present,
            "text_sample": full_text[:500],
        }
    except Exception as exc:
        return {"error": str(exc)}


def _summarize(scenario: str, result: dict) -> dict:
    """Résume un tour : tools, classify, PDF info."""
    events = result["events"]
    ds = result["ds"]

    tools = [f"{ev.get('tool')}.{ev.get('function_name')}"
             for ev in events if ev.get("type") == "tool_call"]
    pdf_path = next((ev.get("output_path") for ev in events
                     if ev.get("type") == "report_ready"), None)
    error_msgs = [ev.get("message", "") for ev in events if ev.get("type") == "error"]
    section_outputs = list((ds.get("section_outputs") or {}).keys())
    sp = ds.get("study_plan") or {}

    summary = {
        "scenario":          scenario,
        "msg":               result["msg"],
        "classify":          result["classify"],
        "tools":             tools,
        "n_tools":           len(tools),
        "section_outputs":   section_outputs,
        "pdf_path":          pdf_path,
        "errors":            error_msgs,
        "data_store_keys":   {
            "write":         ds.get("_write"),
            "report_mode":   ds.get("report_mode"),
            "gender":        sp.get("gender_segmentation"),
            "total_exposure": ds.get("total_exposure"),
            "total_deaths":   ds.get("total_deaths"),
            "qx_table":       len(ds.get("qx_table") or []),
            "section_outputs": len(section_outputs),
        },
    }
    if pdf_path and os.path.exists(pdf_path):
        summary["pdf"] = _analyze_pdf(pdf_path)
    return summary


SCENARIOS = [
    ("S01", "descriptif unisex sans rapport",
     ["fais l'analyse descriptive unisex sans rapport"]),
    ("S02", "descriptif unisex + rapport",
     ["fais un rapport d'analyse descriptive unisex"]),
    ("S03", "descriptif H/F + rapport",
     ["fais un rapport d'analyse descriptive H/F"]),
    ("S04", "taux bruts unisex sans rapport",
     ["calcule les taux bruts unisex sans rapport"]),
    ("S05", "taux bruts unisex + rapport",
     ["fais un rapport avec les taux bruts unisex"]),
    ("S06", "taux bruts H/F + rapport",
     ["construis le rapport avec les taux bruts H/F"]),
    ("S07", "rapport complet unisex (full_report)",
     ["construis le rapport complet de la table de mortalité unisex avec lissage"]),
    ("S08", "rapport complet H/F (full_report)",
     ["construis le rapport complet de la table de mortalité H/F"]),
    ("S09", "demande ambigue → confirmation",
     ["construis une table de mortalité unisex", "oui"]),
    ("S10", "followup rapport après refus",
     ["fais les calculs descriptifs unisex sans rapport",
      "finalement, fais-moi le rapport PDF"]),
]


def main() -> int:
    results = []
    for sid, label, msgs in SCENARIOS:
        session = f"battery_{sid}_{int(time.time()*1000)}"
        print(f"\n{'━' * 78}")
        print(f"  {sid} — {label}")
        print('━' * 78)
        ds = _bootstrap(session)
        result = None
        for i, msg in enumerate(msgs):
            print(f"  Tour {i+1} : {msg!r}")
            result = _run_turn(session, msg, data_store=ds if i == 0 else None)
            ds = None    # tours suivants : laisse le checkpoint LangGraph gérer

        summary = _summarize(label, result)
        results.append(summary)

        # Affichage court
        c = summary["classify"]
        ds_k = summary["data_store_keys"]
        print(f"    classify   : kind={c.get('kind')} write={c.get('write')} "
              f"mode={c.get('report_mode')} conf={c.get('confidence')}")
        print(f"    gender_llm : {c.get('gender_segmentation')}")
        print(f"    n_tools    : {summary['n_tools']}  tools={summary['tools'][:3]}{'…' if len(summary['tools']) > 3 else ''}")
        print(f"    sections   : {summary['section_outputs']}")
        print(f"    PDF        : {summary['pdf_path']}")
        if "pdf" in summary:
            p = summary["pdf"]
            print(f"    PDF stats  : {p.get('pages')} pages, {p.get('n_words')} mots, "
                  f"sections présentes : {[k for k,v in (p.get('sections') or {}).items() if v]}")
        if summary["errors"]:
            print(f"    ERRORS     : {summary['errors'][:2]}")

    # Save
    out = RUNS / "battery_results.json"
    out.write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False))
    print(f"\n\nRésultats détaillés : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
