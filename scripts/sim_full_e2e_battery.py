"""Battery of E2E scenarios — captures real behavior of the agent.

Goal: detect bugs (loops, missing data, PDF generation failures) BEFORE the
demo. Each scenario writes a JSON log with all events to tmp/test_runs/.

Usage:
    python scripts/sim_full_e2e_battery.py                  # all scenarios
    python scripts/sim_full_e2e_battery.py 1 3              # only 1 and 3
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Charge l'API key
_ENV = (_PROJECT_ROOT / ".env").read_text().strip()
for line in _ENV.splitlines():
    if line.startswith("OPENAI_API_KEY="):
        key = line.split("=", 1)[1].strip().strip('"')
        os.environ["OPENAI_API_KEY"] = key
        break

CSV = _PROJECT_ROOT / "Portefeuille" / "portefeuille_test_1000.csv"
RUNS_DIR = _PROJECT_ROOT / "tmp" / "test_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _print(msg: str = "") -> None:
    print(msg, flush=True)


def _line(c: str = "─", n: int = 78) -> None:
    print(c * n, flush=True)


def _make_session(scenario_id: str) -> str:
    return f"e2e_{scenario_id}_{int(time.time())}"


def _bootstrap_session(session_id: str, df: pd.DataFrame, gender: str = "unisex") -> dict:
    """Crée un MemoryManager + bootstrap un column_mapping confirmé.

    Cela permet de simuler un état utilisateur où la désambiguation a
    déjà été faite (équivalent UI : confirmer le mapping)."""
    from session.memory_manager import MemoryManager
    mm = MemoryManager(session_id)
    mm.load()
    mm.register_dataset(df, csv_filename="portefeuille_test_1000.csv")
    # Bootstrap column_mapping (canonique → CSV)
    mm.state.column_mapping = {
        "date_naissance": "date_naissance",
        "date_entree":    "date_entree",
        "date_sortie":    "date_sortie",
        "cause_sortie":   "cause_sortie",
        "sexe":           "sexe",
    }
    mm.state.column_mapping_confirmed = True
    mm.state.disambiguation_done = True
    mm.state.study_plan.observation_start_date = "2018-01-01"
    mm.state.study_plan.observation_end_date   = "2022-12-31"
    mm.state.study_plan.baseline_regulatory_table = "TH-TF-00-02"
    # On précise gender_segmentation pour skipper la question gender
    sp_dict = mm.state.study_plan.model_dump()
    sp_dict["gender_segmentation"] = gender
    from session.session_state import StudyPlan
    mm.state.study_plan = StudyPlan(**sp_dict)
    mm.save()
    return mm.to_data_store()


def _run_turn(
    session_id: str,
    user_msg: str,
    history: list[dict],
    data_store: dict | None = None,
    max_steps: int = 80,
) -> tuple[list[dict], dict]:
    """Lance un tour stream_agent et capture tous les events."""
    from agents.mortality.agents.graph import stream_agent

    history = list(history) + [{"role": "user", "content": user_msg}]
    events: list[dict] = []
    start = time.time()
    for ev in stream_agent(
        history=history,
        data_store=data_store,
        thread_id=session_id,
    ):
        events.append(ev)
        if len(events) > max_steps:
            events.append({"type": "abort", "reason": f"max_steps={max_steps} reached"})
            break

    # Extraire le dernier message AI pour mise à jour de l'historique
    final_message = None
    for ev in reversed(events):
        if ev.get("type") == "message" and ev.get("content"):
            final_message = ev.get("content")
            break
    if final_message:
        history.append({"role": "assistant", "content": final_message})

    elapsed = time.time() - start
    events.append({"type": "_elapsed_s", "value": round(elapsed, 1)})
    return history, {"events": events, "elapsed_s": elapsed}


def _summarize(turn_log: dict) -> dict:
    """Résumé compact d'un tour pour le dashboard."""
    events = turn_log["events"]
    summary = {
        "elapsed_s": turn_log["elapsed_s"],
        "n_events": len(events),
        "agents_triggered": [],
        "tools_called": [],
        "messages": [],
        "errors": [],
        "report_ready": None,
        "done_count": 0,
        "abort": False,
        "_events_full": events,
    }
    for ev in events:
        t = ev.get("type")
        if t == "agent_switch":
            summary["agents_triggered"].append(ev.get("agent"))
        elif t == "tool_call":
            summary["tools_called"].append(f"{ev.get('tool')}.{ev.get('function_name')}")
        elif t == "message":
            content = ev.get("content", "")
            summary["messages"].append(content[:150])
        elif t == "error":
            summary["errors"].append(ev.get("message", "")[:300])
        elif t == "report_ready":
            summary["report_ready"] = ev.get("output_path")
        elif t == "done":
            summary["done_count"] += 1
        elif t == "abort":
            summary["abort"] = True
    return summary


def _save_run(scenario_id: str, scenario_name: str, history: list, turn_logs: list[dict],
              full_events: list[list] | None = None) -> Path:
    out = {
        "scenario_id":   scenario_id,
        "scenario_name": scenario_name,
        "history":       history,
        "turns":         turn_logs,
        "full_events":   full_events or [],
    }
    path = RUNS_DIR / f"scenario_{scenario_id}.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────────────────────
# Scenarios
# ──────────────────────────────────────────────────────────────────────────

def scenario_01_describe_unisex(df) -> dict:
    """S01 — Analyse descriptive simple, unisex, pas de rapport."""
    sid = "01_describe_unisex_no_report"
    session = _make_session(sid)
    ds = _bootstrap_session(session, df, gender="unisex")

    history: list[dict] = []
    turns: list[dict] = []

    history, turn = _run_turn(session, "Fais-moi une analyse descriptive sans rapport", history, ds)
    turns.append({"user": "Fais-moi une analyse descriptive sans rapport",
                  "summary": _summarize(turn),
                  "_events_full": turn.get("events", [])})

    return {"id": sid, "session": session, "turns": turns, "history": history}


def scenario_02_describe_unisex_with_report(df) -> dict:
    """S02 — Analyse descriptive avec rapport PDF."""
    sid = "02_describe_unisex_with_report"
    session = _make_session(sid)
    ds = _bootstrap_session(session, df, gender="unisex")

    history: list[dict] = []
    turns: list[dict] = []

    history, turn = _run_turn(session, "Fais-moi un rapport descriptif de mon portefeuille", history, ds)
    turns.append({"user": "Fais-moi un rapport descriptif de mon portefeuille",
                  "summary": _summarize(turn)})

    return {"id": sid, "session": session, "turns": turns, "history": history}


def scenario_03_describe_by_sex_with_report(df) -> dict:
    """S03 — Analyse descriptive par sexe avec rapport."""
    sid = "03_describe_by_sex_with_report"
    session = _make_session(sid)
    ds = _bootstrap_session(session, df, gender="by_sex")

    history: list[dict] = []
    turns: list[dict] = []

    history, turn = _run_turn(session, "Fais-moi un rapport descriptif H/F", history, ds)
    turns.append({"user": "Fais-moi un rapport descriptif H/F",
                  "summary": _summarize(turn)})

    return {"id": sid, "session": session, "turns": turns, "history": history}


def scenario_04_construct_table_ambiguous(df) -> dict:
    """S04 — Demande ambigue, doit poser la question rapport."""
    sid = "04_construct_table_ambiguous"
    session = _make_session(sid)
    ds = _bootstrap_session(session, df, gender="unisex")

    history: list[dict] = []
    turns: list[dict] = []

    history, turn = _run_turn(session, "Construis-moi une table de mortalité", history, ds)
    turns.append({"user": "Construis-moi une table de mortalité",
                  "summary": _summarize(turn)})

    history, turn = _run_turn(session, "oui, fais le rapport", history, None)
    turns.append({"user": "oui, fais le rapport",
                  "summary": _summarize(turn)})

    return {"id": sid, "session": session, "turns": turns, "history": history}


def scenario_05_question_only(df) -> dict:
    """S05 — Question conversationnelle, pas de tools."""
    sid = "05_question_only"
    session = _make_session(sid)
    ds = _bootstrap_session(session, df, gender="unisex")

    history: list[dict] = []
    turns: list[dict] = []

    history, turn = _run_turn(session, "C'est quoi le lissage Whittaker-Henderson ?", history, ds)
    turns.append({"user": "C'est quoi le lissage Whittaker-Henderson ?",
                  "summary": _summarize(turn)})

    return {"id": sid, "session": session, "turns": turns, "history": history}


def scenario_06_explicit_no_report(df) -> dict:
    """S06 — User dit explicitement pas de rapport."""
    sid = "06_explicit_no_report"
    session = _make_session(sid)
    ds = _bootstrap_session(session, df, gender="unisex")

    history: list[dict] = []
    turns: list[dict] = []

    history, turn = _run_turn(session, "Calcule l'analyse descriptive sans rapport", history, ds)
    turns.append({"user": "Calcule l'analyse descriptive sans rapport",
                  "summary": _summarize(turn)})

    return {"id": sid, "session": session, "turns": turns, "history": history}


def scenario_07_raw_rates_request(df) -> dict:
    """S07 — Demande explicite de taux bruts (mode raw_rates)."""
    sid = "07_raw_rates_request"
    session = _make_session(sid)
    ds = _bootstrap_session(session, df, gender="unisex")

    history: list[dict] = []
    turns: list[dict] = []

    history, turn = _run_turn(session, "Fais-moi un rapport avec les taux bruts seuls", history, ds)
    turns.append({"user": "Fais-moi un rapport avec les taux bruts seuls",
                  "summary": _summarize(turn)})

    return {"id": sid, "session": session, "turns": turns, "history": history}


def scenario_08_smoothed_rates(df) -> dict:
    """S08 — Demande de taux lissés."""
    sid = "08_smoothed_rates"
    session = _make_session(sid)
    ds = _bootstrap_session(session, df, gender="unisex")

    history: list[dict] = []
    turns: list[dict] = []

    history, turn = _run_turn(session, "Fais-moi un rapport avec les taux lissés", history, ds)
    turns.append({"user": "Fais-moi un rapport avec les taux lissés",
                  "summary": _summarize(turn)})

    return {"id": sid, "session": session, "turns": turns, "history": history}


def scenario_09_followup_report_after_no(df) -> dict:
    """S09 — Calculs faits sans rapport, puis user demande rapport."""
    sid = "09_followup_report_after_no"
    session = _make_session(sid)
    ds = _bootstrap_session(session, df, gender="unisex")

    history: list[dict] = []
    turns: list[dict] = []

    history, turn = _run_turn(session, "Fais l'analyse descriptive sans rapport", history, ds)
    turns.append({"user": "Fais l'analyse descriptive sans rapport",
                  "summary": _summarize(turn)})

    history, turn = _run_turn(session, "Finalement, fais-moi le rapport PDF", history, None)
    turns.append({"user": "Finalement, fais-moi le rapport PDF",
                  "summary": _summarize(turn)})

    return {"id": sid, "session": session, "turns": turns, "history": history}


def scenario_10_unclear_input(df) -> dict:
    """S10 — Demande très vague pour stresser la classification."""
    sid = "10_unclear_input"
    session = _make_session(sid)
    ds = _bootstrap_session(session, df, gender="unisex")

    history: list[dict] = []
    turns: list[dict] = []

    history, turn = _run_turn(session, "Vas-y", history, ds)
    turns.append({"user": "Vas-y",
                  "summary": _summarize(turn)})

    return {"id": sid, "session": session, "turns": turns, "history": history}


SCENARIOS = [
    ("01", scenario_01_describe_unisex,                  "Analyse descriptive (sans rapport)"),
    ("02", scenario_02_describe_unisex_with_report,      "Analyse descriptive + rapport"),
    ("03", scenario_03_describe_by_sex_with_report,      "Analyse descriptive H/F + rapport"),
    ("04", scenario_04_construct_table_ambiguous,        "Demande ambigue → confirmation rapport"),
    ("05", scenario_05_question_only,                    "Question conversationnelle"),
    ("06", scenario_06_explicit_no_report,               "Refus explicite du rapport"),
    ("07", scenario_07_raw_rates_request,                "Taux bruts (mode raw_rates)"),
    ("08", scenario_08_smoothed_rates,                   "Taux lissés"),
    ("09", scenario_09_followup_report_after_no,         "Rapport demandé après refus initial"),
    ("10", scenario_10_unclear_input,                    "Demande vague"),
]


def run_one(num: str, df, fn, name: str) -> dict:
    _print()
    _line("█")
    _print(f"█  SCÉNARIO {num} — {name}")
    _line("█")
    try:
        result = fn(df)
        full = [t.get("_events_full") for t in result["turns"]]
        path = _save_run(result["id"], name, result["history"], result["turns"], full)
        _print(f"  → log saved : {path}")
        # Affichage compact
        for i, t in enumerate(result["turns"], 1):
            s = t["summary"]
            _print(f"  Turn {i} | user: {t['user']!r}")
            _print(f"    elapsed     : {s['elapsed_s']:.1f}s")
            _print(f"    agents      : {' → '.join(s['agents_triggered'])}")
            _print(f"    tools       : {s['tools_called'][:8]}")
            for m in s["messages"][:5]:
                _print(f"    msg         : {m}")
            for e in s["errors"]:
                _print(f"    ERROR       : {e}")
            if s["report_ready"]:
                _print(f"    PDF         : {s['report_ready']}")
            if s["abort"]:
                _print(f"    ABORTED")
        return {"id": num, "ok": True, "result": result}
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        _print(f"  ❌ EXCEPTION : {exc}")
        _print(tb)
        return {"id": num, "ok": False, "error": str(exc), "tb": tb}


def main(argv: list[str]) -> int:
    df = pd.read_csv(CSV)
    _print(f"Loaded CSV: {len(df)} rows × {len(df.columns)} cols")

    selected = SCENARIOS
    if len(argv) > 1:
        wanted = set(argv[1:])
        selected = [s for s in SCENARIOS if s[0] in wanted]

    results = []
    for num, fn, name in selected:
        results.append(run_one(num, df, fn, name))

    _print()
    _line("═")
    _print(f"  Battery complete — {len(results)} scenarios")
    _line("═")
    n_ok = sum(1 for r in results if r["ok"])
    _print(f"  OK     : {n_ok}/{len(results)}")
    _print(f"  Errors : {len(results) - n_ok}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
