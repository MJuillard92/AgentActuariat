"""Niveau 3 — Master → Builder réel sur CSV (mode description, moins coûteux).

Charge le CSV de test, simule la désambiguation (déjà faite), envoie un
message "fais-moi un rapport descriptif" (write=yes, mode=description) et
laisse Master + Builder tourner pour de vrai.

S'arrête après <BUILD_DONE> ou max 8 tours total Builder pour rester dans
le budget.

Usage :
    python scripts/sim_test_master_builder.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from io import StringIO

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")


# ──────────────────────────────────────────────────────────────────────────
# Setup : charger le CSV + créer dataset_ref
# ──────────────────────────────────────────────────────────────────────────

CSV_PATH = _PROJECT_ROOT / "Portefeuille" / "portefeuille_test_1000.csv"
SESSION_ID = "sim_test_session_xyz"


def _setup_dataset_store():
    """Charge le CSV et le persiste via DatasetStore avec un session_id fixe."""
    from session.dataset_store import DatasetStore

    df = pd.read_csv(CSV_PATH)
    print(f"  ✓ CSV chargé : {len(df)} lignes, colonnes = {list(df.columns)}")

    meta = DatasetStore.store(SESSION_ID, df)
    print(f"  ✓ Dataset stocké : {meta}")
    return df


# ──────────────────────────────────────────────────────────────────────────
# Pretty-print
# ──────────────────────────────────────────────────────────────────────────

def _short(s, n=160):
    s = str(s or "")
    return s if len(s) <= n else s[:n] + "…"


def _print_step(label, out):
    print(f"\n  ── {label} ──")
    msgs = out.get("messages") or []
    for m in msgs:
        cls = type(m).__name__
        content = (getattr(m, "content", "") or "")
        tcs = getattr(m, "tool_calls", None) or []
        if tcs:
            print(f"     {cls} (tool_calls=[{', '.join(tc.get('name', '?') for tc in tcs)}])")
        else:
            print(f"     {cls} : {_short(content, 200)}")
    if out.get("active_agent"):
        print(f"     → active_agent = {out['active_agent']}")
    events = out.get("events") or []
    for ev in events:
        if isinstance(ev, dict):
            etype = ev.get("type")
            if etype in ("done", "agent_switch"):
                print(f"     event: {etype} | {ev.get('agent') or ''}")
            elif etype == "message":
                print(f"     event: message | {_short(ev.get('content'), 120)}")


# ──────────────────────────────────────────────────────────────────────────
# Main : un parcours Master → Builder
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("┌" + "─" * 76 + "┐")
    print("│  Niveau 3 — Master → Builder réel (mode description, ~0,15 €)            │")
    print("└" + "─" * 76 + "┘")

    print("\n[1] Chargement du CSV + DatasetStore")
    _setup_dataset_store()

    print("\n[2] Construction de l'état initial")
    initial_state = {
        "messages": [],
        "data_store": {
            # Désambiguation déjà faite : colonnes et valeurs canoniques
            "_disambiguation_done":     True,
            "column_mapping_confirmed": True,
            "value_mapping_confirmed":  True,
            "records_normalized":       True,
            "study_plan":               {"gender_segmentation": "unisex"},
            # Charger les input_records pour que les tools statistical_analysis trouvent le df
            "input_records":            None,  # sera rempli via dataset_ref
        },
        "dataset_ref": SESSION_ID,
        "context_docs": [],
    }

    # Ajouter le DataFrame normalisé directement dans data_store
    df = pd.read_csv(CSV_PATH)
    initial_state["data_store"]["input_records"] = df

    print(f"  ✓ State initialisé. session_id={SESSION_ID}, dataset_ref={SESSION_ID}")
    print(f"  ✓ DataFrame dans data_store : {len(df)} lignes")

    print("\n[3] Tour 1 — User envoie : 'fais-moi un rapport descriptif'")
    state = dict(initial_state)
    state["messages"].append(HumanMessage(content="construit les taux bruts et le rapport associé"))

    from agents.mortality.agents import master_node as mn

    t0 = time.time()
    out = mn.master_node(state)
    print(f"  ⏱  {time.time() - t0:.1f} s")
    _print_step("master_node sortie", out)

    # Appliquer les updates
    state["messages"].extend(out.get("messages") or [])
    if "data_store" in out:
        state["data_store"] = out["data_store"]
    state["active_agent"] = out.get("active_agent")

    if state.get("active_agent") != "builder":
        print("\n  ⚠  Master n'a pas routé vers Builder. Fin du test.")
        return 0

    print("\n[4] Tours Builder (max 5 tours LLM)")
    from agents.mortality.agents import builder_node as bn
    from agents.mortality.agents.graph import _should_continue_builder
    from agents.mortality.agents.tools_node import execute_tools

    n_llm_calls = 0
    max_llm_calls = 5
    while n_llm_calls < max_llm_calls:
        n_llm_calls += 1
        print(f"\n  ── Builder LLM call {n_llm_calls} ──")
        t0 = time.time()
        b_out = bn.builder_node(state)
        print(f"     ⏱  {time.time() - t0:.1f} s")
        _print_step(f"builder_node tour {n_llm_calls}", b_out)

        # Appliquer
        state["messages"].extend(b_out.get("messages") or [])
        if "data_store" in b_out:
            state["data_store"] = b_out["data_store"]

        # Verdict de routing
        decision = _should_continue_builder(state)
        print(f"     → _should_continue_builder = {decision!r}")

        if decision == "to_master":
            print("\n  ✓ Builder a rendu la main au Master.")
            break
        if decision == "tools":
            # Exécuter les tools via le helper du projet
            print("     → Exécution des tools…")
            try:
                t0 = time.time()
                tool_out = execute_tools(state)
                print(f"     ⏱  tools : {time.time() - t0:.1f} s")
                state["messages"].extend(tool_out.get("messages") or [])
                if "data_store" in tool_out:
                    state["data_store"] = tool_out["data_store"]
                last = state["messages"][-1] if state["messages"] else None
                if last:
                    name = getattr(last, "name", "?")
                    print(f"     ToolMessage de {name} : {_short(getattr(last, 'content', ''), 150)}")
            except Exception as exc:
                print(f"     ⚠  Erreur tool : {exc}")
                break
        else:  # END
            print(f"     ↺ Sortie via {decision}, on s'arrête.")
            break

    print("\n[4 bis] Tour 2 — Réponse user 'ok' réinjectée DIRECTEMENT vers Builder")
    print("      (on contourne Master pour ne pas reclassifier — bug à corriger plus tard)")
    # Si Builder s'est arrêté avec une présentation, on simule la réponse user
    # et on relance directement Builder pour qu'il appelle les tools
    if state.get("active_agent") == "builder":
        state["messages"].append(HumanMessage(content="ok lance les calculs maintenant"))

        if True:  # bypass Master
            n_more_calls = 0
            max_more = 4
            while n_more_calls < max_more:
                n_more_calls += 1
                print(f"\n  ── Builder LLM call (suite) {n_more_calls} ──")
                t0 = time.time()
                b_out = bn.builder_node(state)
                print(f"     ⏱  {time.time() - t0:.1f} s")
                _print_step(f"builder tour {n_more_calls}", b_out)
                state["messages"].extend(b_out.get("messages") or [])
                if "data_store" in b_out:
                    state["data_store"] = b_out["data_store"]
                decision = _should_continue_builder(state)
                print(f"     → _should_continue_builder = {decision!r}")
                if decision == "to_master":
                    break
                if decision == "tools":
                    try:
                        t0 = time.time()
                        tool_out = execute_tools(state)
                        print(f"     ⏱  tools : {time.time() - t0:.1f} s")
                        state["messages"].extend(tool_out.get("messages") or [])
                        if "data_store" in tool_out:
                            state["data_store"] = tool_out["data_store"]
                        last = state["messages"][-1] if state["messages"] else None
                        if last:
                            name = getattr(last, "name", "?")
                            print(f"     ToolMessage de {name} : {_short(getattr(last, 'content', ''), 150)}")
                    except Exception as exc:
                        print(f"     ⚠  Erreur tool : {exc}")
                        break
                else:
                    break

    print("\n[5] État final")
    ds = state.get("data_store") or {}
    builder_outputs_keys = (
        "cleaned_records", "exclusion_report", "total_records",
        "total_exposure", "total_deaths", "segmentations",
        "serie", "serie_h", "serie_f", "ages",
    )
    print(f"  → builder_outputs présentes :")
    for k in builder_outputs_keys:
        present = k in ds and ds[k] is not None
        marker = "✓" if present else "✗"
        if present:
            v = ds[k]
            preview = str(v)[:60] if not isinstance(v, list) else f"list[{len(v)}]"
            print(f"     {marker} {k:30s} → {preview}")
        else:
            print(f"     {marker} {k}")

    print(f"\n  → _master_builder_cycles = {ds.get('_master_builder_cycles', 0)}")
    print(f"  → _builder_turns          = {ds.get('_builder_turns', 0)}")
    print(f"  → n_llm_calls (Builder)   = {n_llm_calls}")
    print(f"  → active_agent final      = {state.get('active_agent')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
