"""Niveau 4 — Writer pipeline réel.

Reprend l'état final du Niveau 3 (data_store rempli avec les 8 clés Bloc A)
et lance le pipeline Writer : load_plan → completion → redaction → assemblage.

Coût estimé : ~0,15 € (3 sections × rédaction LLM + 1 RAG par section).

Usage :
    python scripts/sim_test_writer.py
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")


CSV_PATH = _PROJECT_ROOT / "Portefeuille" / "portefeuille_test_1000.csv"


def _build_data_store() -> dict:
    """Construit un data_store complet équivalent au résultat Niveau 3."""
    df = pd.read_csv(CSV_PATH)

    # Lance les tools réels pour obtenir des données cohérentes
    print("  → preprocessing.clean_records …")
    from tools.preprocessing.clean_records import run as run_clean
    cleaned = run_clean(df)

    print("  → builder.exposure …")
    from tools.builder.exposure import run as run_expo
    expo = run_expo(df, {"age_min": 0, "age_max": 100, "observation_end": "31/12/2024"})

    print("  → statistical_analysis.segmentation …")
    from tools.statistical_analysis.segmentation import run as run_seg
    seg = run_seg(df)

    print("  → statistical_analysis.time_series …")
    from tools.statistical_analysis.time_series import run as run_ts
    ts = run_ts(df, {})

    print("  → statistical_analysis.age_distribution …")
    from tools.statistical_analysis.age_distribution import run as run_age
    ages = run_age(df, {})

    final_count = (cleaned.get("exclusion_report") or {}).get("final_count", len(df))

    return {
        # Master_from_data — inférés
        "study_objective":        "construction_table_mortalite",
        "start_year":             2018,
        "end_year":               2023,
        "num_observation_years":  6,
        # Bloc A — preprocessing
        "cleaned_records":  cleaned.get("cleaned_records"),
        "exclusion_report": cleaned.get("exclusion_report"),
        "total_records":    final_count,
        # Bloc B — exposure
        "total_exposure":   expo.get("total_exposure", 0),
        "total_deaths":     expo.get("total_deaths", 0),
        "exposure_table":   expo.get("exposure_table", []),
        # Bloc A — stats
        "segmentations":    seg.get("segmentations", {}),
        "serie":            ts.get("serie", []),
        "ages":             ages,
        # Mode + study_plan
        "report_mode":      "description",
        "study_plan":       {"gender_segmentation": "unisex"},
    }


def main() -> int:
    print("┌" + "─" * 76 + "┐")
    print("│  Niveau 4 — Writer pipeline (mode description, ~0,15 €)                  │")
    print("└" + "─" * 76 + "┘")

    print("\n[1] Construction d'un data_store complet")
    data_store = _build_data_store()
    def _is_present(v):
        if v is None:
            return False
        if isinstance(v, pd.DataFrame):
            return not v.empty
        if isinstance(v, (list, dict, str)):
            return len(v) > 0
        return True
    keys_present = sum(1 for v in data_store.values() if _is_present(v))
    print(f"  ✓ data_store : {keys_present} clés non vides")

    print("\n[2] Étape 1 — load_plan (Design 3, lecture YAML)")
    from agents.report.pipeline._01_load_plan import load_plan
    t0 = time.time()
    plan = load_plan(
        data_store,
        study_plan=data_store.get("study_plan"),
        context={"report_mode": "description", "gender_segmentation": "unisex"},
    )
    print(f"  ⏱  {time.time() - t0:.1f} s")
    print(f"  ✓ {plan.n_ready}/{plan.n_total} sections prêtes")
    for sec in plan.sections:
        marker = "✓" if sec.ready else "✗"
        n_visuals = len(sec.visual_specs)
        miss = sec.missing_inputs[:3] if sec.missing_inputs else []
        suffix = f" (manque : {miss}…)" if miss else ""
        print(f"     {marker} {sec.section_id:25s} | {n_visuals} visuels{suffix}")

    if plan.missing_fields:
        print(f"  ⚠ Clés globalement manquantes : {plan.missing_fields[:5]}")

    print("\n[3] Étape 2 — completion_plan (RAG par section)")
    from agents.report.pipeline._03_completion_plan import complete_plan
    t0 = time.time()
    plan = complete_plan(plan, data_store=data_store)
    print(f"  ⏱  {time.time() - t0:.1f} s")
    print(f"  ✓ Plan enrichi avec contenus RAG")

    print("\n[4] Étape 3 — _04_redaction (rédaction LLM par section)")
    from agents.report.pipeline._04_redaction import redact_plan
    t0 = time.time()
    data_store = redact_plan(plan, data_store=data_store)  # mis à jour en place + retourné
    print(f"  ⏱  {time.time() - t0:.1f} s")
    section_outputs = data_store.get("section_outputs") or {}
    print(f"  ✓ {len(section_outputs)} section_outputs produites")

    for sid, output in section_outputs.items():
        text = (output or {}).get("redacted_text") or (output or {}).get("text") or ""
        skipped = (output or {}).get("skipped", False)
        if skipped:
            print(f"\n  ── {sid} — SKIPPED ──")
            print(f"     reason : {(output or {}).get('reason', '')}")
            continue
        n_chars = len(text)
        preview = text[:200].replace("\n", " ")
        print(f"\n  ── {sid} — {n_chars} caractères ──")
        print(f"     {preview}…")

    print("\n[5] Persistance du résultat")
    out_path = _PROJECT_ROOT / "tmp" / "sim_test_writer_output.json"
    out_path.parent.mkdir(exist_ok=True)
    serializable = {
        "n_sections": len(section_outputs),
        "sections":   {
            sid: {
                "skipped":       (out or {}).get("skipped", False),
                "redacted_text": (out or {}).get("redacted_text") or (out or {}).get("text"),
            } for sid, out in section_outputs.items()
        },
    }
    out_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2, default=str))
    print(f"  ✓ Écrit : {out_path}")

    print("\n" + "═" * 78)
    print(f"  Niveau 4 terminé. {len(section_outputs)} sections traitées.")
    print(f"  Vérifie le fichier de sortie : {out_path}")
    print("═" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
