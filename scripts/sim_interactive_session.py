"""Session interactive avec l'agent réel.

Usage : python scripts/sim_interactive_session.py --session SID --msg "..."
        python scripts/sim_interactive_session.py --session SID --reset

Garde la session vivante entre les appels via SessionState sur disque
+ MemorySaver LangGraph (le thread_id = session_id).
"""
from __future__ import annotations

import argparse
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


SESSIONS_DIR = _PROJECT_ROOT / "session" / "data"
INTER_DIR = _PROJECT_ROOT / "tmp" / "interactive"
INTER_DIR.mkdir(parents=True, exist_ok=True)


def _history_path(sid: str) -> Path:
    return INTER_DIR / f"{sid}_history.json"


def _load_history(sid: str) -> list:
    p = _history_path(sid)
    if p.exists():
        return json.loads(p.read_text())
    return []


def _save_history(sid: str, history: list) -> None:
    _history_path(sid).write_text(json.dumps(history, indent=2, ensure_ascii=False))


def _bootstrap(sid: str) -> None:
    """Bootstrap minimal : column_mapping confirmé, RIEN d'autre."""
    from session.memory_manager import MemoryManager
    mm = MemoryManager(sid); mm.load()
    if mm.state.dataset_meta is None:
        df = pd.read_csv(_PROJECT_ROOT / "Portefeuille" / "portefeuille_test_1000.csv")
        mm.register_dataset(df, csv_filename="portefeuille_test_1000.csv")
        mm.state.column_mapping = {
            "date_naissance": "date_naissance", "date_entree": "date_entree",
            "date_sortie":    "date_sortie",    "cause_sortie": "cause_sortie",
            "sexe":           "sexe",
        }
        mm.state.column_mapping_confirmed = True
        mm.state.disambiguation_done = True
        mm.save()


def _reset(sid: str) -> None:
    for f in SESSIONS_DIR.glob(f"{sid}_*"):
        f.unlink(missing_ok=True)
    for f in (SESSIONS_DIR / "artifacts").glob(f"{sid}_*"):
        f.unlink(missing_ok=True)
    _history_path(sid).unlink(missing_ok=True)
    print(f"Session {sid} reset.")


def _run_turn(sid: str, user_msg: str) -> dict:
    """Lance un tour, capture classify + events. Retourne un dict structuré."""
    from session.memory_manager import MemoryManager
    from agents.mortality.agents import master_node as mn
    from agents.mortality.agents.graph import stream_agent

    _bootstrap(sid)

    # Intercepter classify_intent pour capturer son retour
    orig = mn._classify_intent
    captured = {}
    def _spy(last_human, data_store, dataset_ref):
        r = orig(last_human, data_store, dataset_ref)
        captured[last_human] = dict(r)
        return r
    mn._classify_intent = _spy

    history = _load_history(sid)
    history.append({"role": "user", "content": user_msg})

    mm = MemoryManager(sid); mm.load()
    events = []
    for ev in stream_agent(
        history=history,
        data_store=mm.to_data_store(),
        thread_id=sid,
    ):
        events.append(ev)

    # Récupérer dernière réponse assistant
    last_ai = next((e.get("content") for e in reversed(events)
                    if e.get("type") == "message" and e.get("content")), "")
    history.append({"role": "assistant", "content": last_ai})
    _save_history(sid, history)

    # État final
    mm2 = MemoryManager(sid); mm2.load()
    final_ds = mm2.to_data_store()

    return {
        "user_msg":   user_msg,
        "classify":   captured.get(user_msg, {}),
        "events":     events,
        "final_ds":   final_ds,
        "last_ai":    last_ai,
    }


def _format_report(r: dict) -> str:
    cls = r["classify"]
    events = r["events"]
    ds = r["final_ds"]

    # Tools
    tools = []
    for ev in events:
        if ev.get("type") == "tool_call":
            tools.append(f"{ev.get('tool')}.{ev.get('function_name')}")

    # Agents traversés
    agents = []
    last = None
    for ev in events:
        if ev.get("type") == "agent_switch":
            ag = ev.get("agent")
            if ag != last:
                agents.append(ag)
                last = ag

    # Messages bot
    msgs = [ev.get("content", "") for ev in events
            if ev.get("type") == "message" and ev.get("content")]

    # PDF ?
    pdf = next((ev.get("output_path") for ev in events
                if ev.get("type") == "report_ready"), None)

    # Décision Master (deviner depuis events + ds)
    decision = "—"
    if any("reformuler" in m.lower() for m in msgs):
        decision = "demande de reformulation (confidence < seuil)"
    elif any("Voulez-vous que je génère un rapport PDF" in m for m in msgs):
        decision = "pose la question PDF (write=ask)"
    elif any("unisex" in m.lower() and "agrégée" in m.lower() for m in msgs):
        decision = "pose la question gender (unisex / by_sex)"
    elif "WriterAgent" in agents and tools:
        decision = "route Builder puis Writer → PDF"
    elif "WriterAgent" in agents and not tools:
        decision = "route DIRECT vers Writer (data_store déjà complet)"
    elif "BuilderAgent" in agents and not pdf:
        decision = "route Builder → calculs, pas de rapport"
    elif "MasterAgent" in agents and not tools and not pdf:
        decision = "réponse conversationnelle (kind=question) ou demande reformulation"

    out = []
    out.append("─" * 78)
    out.append(f"USER : {r['user_msg']!r}")
    out.append("─" * 78)
    out.append("\nCLASSIFY :")
    out.append(f"   kind        = {cls.get('kind')!r}")
    out.append(f"   write       = {cls.get('write')!r}")
    out.append(f"   report_mode = {cls.get('report_mode')!r}")
    out.append(f"   confidence  = {cls.get('confidence')}")
    out.append(f"   reasoning   = {cls.get('reasoning')!r}")
    out.append(f"   reply       = {cls.get('reply')!r}")

    out.append(f"\nDÉCISION MASTER : {decision}")
    out.append(f"   chemin agents : {' → '.join(agents) if agents else '—'}")

    out.append("\nTOOLS appelés :")
    if tools:
        for t in tools:
            out.append(f"   - {t}")
    else:
        out.append("   (aucun)")

    out.append("\nDATA_STORE (extraits) :")
    interesting = [
        "_kind", "_write", "report_mode",
        "_write_question_asked", "_reformulation_attempts",
        "_master_builder_cycles",
        "gender_segmentation",   # déduplique avec sp
        "total_exposure", "total_deaths", "total_records",
    ]
    for k in interesting:
        v = ds.get(k)
        if v is not None:
            out.append(f"   {k:30} = {v}")
    sp = ds.get("study_plan") or {}
    if sp.get("gender_segmentation"):
        out.append(f"   {'study_plan.gender_segmentation':30} = {sp.get('gender_segmentation')!r}")
    so = ds.get("section_outputs") or {}
    if so:
        out.append(f"   section_outputs : {list(so.keys())}")

    if pdf:
        out.append(f"\nPDF : {pdf}")

    out.append("\nRÉPONSE AGENT (visible par user) :")
    if r["last_ai"]:
        for line in r["last_ai"].split("\n")[:15]:
            out.append(f"   │ {line}")
    out.append("─" * 78)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--msg", default=None)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--state", action="store_true")
    args = ap.parse_args()

    if args.reset:
        _reset(args.session)
        return

    if args.state:
        from session.memory_manager import MemoryManager
        mm = MemoryManager(args.session); mm.load()
        ds = mm.to_data_store()
        print(json.dumps({k: str(v)[:200] for k, v in ds.items()}, indent=2, ensure_ascii=False))
        return

    if not args.msg:
        ap.error("Need --msg or --reset or --state")

    r = _run_turn(args.session, args.msg)
    print(_format_report(r))


if __name__ == "__main__":
    main()
