"""Simulation end-to-end du pattern Builder→Master question delegation.

3 scénarios :
  A. study_plan déjà rempli → Niveau 1, 0 LLM call.
  B. Signal implicite dans le message user → Niveau 2 (LLM mini).
  C. Pas de signal → Niveau 3 (forward), simulation user répond, Master extrait.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
load_dotenv(_PROJECT_ROOT / ".env")


def _builder_emits_lambda_question():
    return AIMessage(
        content="J'ai besoin d'une précision pour le lissage.",
        additional_kwargs={
            "need_user_input": {
                "context_key": "smoothing_lambda",
                "question":    "Quel paramètre lambda Whittaker ? (100=souple, 200=standard, 500=fort)",
                "options":     [100, 200, 500],
                "default":     200,
            }
        }
    )


def _print_outcome(label, out):
    print(f"\n  ── {label} ──")
    print(f"     active_agent      : {out.get('active_agent')}")
    msgs = out.get("messages") or []
    for m in msgs:
        cls = type(m).__name__
        c = (getattr(m, "content", "") or "")[:200]
        print(f"     {cls}: {c}")
    sp = (out.get("data_store") or {}).get("study_plan") or {}
    if sp:
        print(f"     study_plan        : {sp}")


def scenario_a_level1():
    from agents.mortality.agents import master_node as mn
    print("\n" + "█" * 70)
    print("  SCÉNARIO A — Niveau 1 (study_plan rempli)")
    print("█" * 70)
    state = {
        "messages":    [HumanMessage(content="construit la table"),
                        _builder_emits_lambda_question()],
        "data_store":  {
            "_disambiguation_done":   True,
            "_master_builder_cycles": 1,
            "_user_messages":         ["construit la table"],
            "study_plan":             {"smoothing_lambda": 300, "gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
        "active_agent": "master",
    }
    out = mn.master_node(state)
    _print_outcome("Master", out)
    assert out["data_store"]["study_plan"]["smoothing_lambda"] == 300
    print("  ✓ Niveau 1 OK — réponse 300 injectée sans appel LLM.")


def scenario_b_level2():
    from agents.mortality.agents import master_node as mn
    print("\n" + "█" * 70)
    print("  SCÉNARIO B — Niveau 2 (LLM infère depuis 'lissage doux')")
    print("█" * 70)
    state = {
        "messages":    [HumanMessage(content="construit la table avec un lissage doux"),
                        _builder_emits_lambda_question()],
        "data_store":  {
            "_disambiguation_done":   True,
            "_master_builder_cycles": 1,
            "_user_messages":         ["construit la table avec un lissage doux"],
            "study_plan":             {"gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
        "active_agent": "master",
    }
    out = mn.master_node(state)
    _print_outcome("Master", out)
    sp_value = out["data_store"]["study_plan"].get("smoothing_lambda")
    print(f"  → Mini a inféré : {sp_value}")
    if sp_value == 100:
        print("  ✓ Niveau 2 OK — 'doux' → 100.")
    elif sp_value is not None:
        print(f"  ✓ Niveau 2 a inféré une valeur ({sp_value}) — vérifier la cohérence métier.")
    else:
        print("  ⚠ Niveau 2 a forward (LLM pas confiant). C'est conservateur, OK.")


def scenario_c_level3():
    from agents.mortality.agents import master_node as mn
    print("\n" + "█" * 70)
    print("  SCÉNARIO C — Niveau 3 (forward + extract)")
    print("█" * 70)

    state = {
        "messages":    [HumanMessage(content="construit la table"),
                        _builder_emits_lambda_question()],
        "data_store":  {
            "_disambiguation_done":   True,
            "_master_builder_cycles": 1,
            "_user_messages":         ["construit la table"],
            "study_plan":             {"gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
        "active_agent": "master",
    }
    out = mn.master_node(state)
    _print_outcome("Master tour 1 (forward)", out)
    if "_pending_need" not in out["data_store"]:
        print("  ⚠ Niveau 2 a inféré la valeur sans demander à l'user — scénario non testé.")
        return
    state["data_store"] = out["data_store"]
    state["messages"].extend(out.get("messages") or [])

    print("\n  → User répond : '500 ça me va'")
    state["messages"].append(HumanMessage(content="500 ça me va"))
    out2 = mn.master_node(state)
    _print_outcome("Master tour 2 (extract + route)", out2)
    final_value = out2["data_store"]["study_plan"].get("smoothing_lambda")
    if final_value == 500:
        print("  ✓ Niveau 3 OK — réponse 500 capturée et cachée.")
    else:
        print(f"  ⚠ Valeur finale : {final_value} (500 attendu).")
    assert "_pending_need" not in out2["data_store"]
    assert out2.get("active_agent") == "builder"


def main():
    print("┌" + "─" * 68 + "┐")
    print("│  Simulation : Builder→Master question delegation                 │")
    print("│  Coût estimé : ~0,005 €                                          │")
    print("└" + "─" * 68 + "┘")
    scenario_a_level1()
    scenario_b_level2()
    scenario_c_level3()
    print("\n" + "═" * 70)
    print("  Simulation terminée.")
    print("═" * 70)


if __name__ == "__main__":
    main()
