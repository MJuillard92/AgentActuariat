"""
run_pipeline.py
---------------
Lance le pipeline complet de construction de table de mortalité en mode autonome :
  exposure → crude_rates → diagnostics → smoothing → validation → benchmarking → certification PDF

Toutes les interactions de l'agent (tool_calls, tool_results, messages) sont loggées
dans log_temp.json à la racine du projet.

Usage :
    python run_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

# Racine du projet
ROOT = Path(__file__).parent

# Charger les variables d'environnement depuis .env si présent
_env_file = ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())
DATA_FILE = ROOT / "uploads" / "données_actif_decede.txt"
LOG_FILE  = ROOT / "log_temp.json"
PDF_OUT   = "/tmp/rapport_pipeline_test.pdf"

PROMPT = (
    "Lance une analyse complète de construction de table de mortalité sur le portefeuille chargé. "
    "Utilise EXACTEMENT ces tools dans cet ordre, avec ces paramètres :\n"
    "1. builder (function_name=exposure, params={})\n"
    "2. builder (function_name=crude_rates, params={method: central})\n"
    "3. builder (function_name=diagnostics, params={function_name: credibility})\n"
    "4. builder (function_name=smoothing, params={method: whittaker, lambda_wh: 1000})\n"
    "   → Si n_non_monotone > 0 : relancer smoothing avec lambda_wh doublé jusqu'à monotonie\n"
    "5. builder (function_name=validation, params={function_name: confidence_intervals})\n"
    "6. builder (function_name=benchmarking, params={function_name: abatement_factors, sexe: H, reference_name: TH0002})\n"
    f"7. build_pdf (function_name=certification_report, params={{output_path: {PDF_OUT}, sexe: H}})\n"
    "IMPORTANT : procède de façon AUTONOME. Ne pose pas de questions. "
    "Les noms de functions ci-dessus sont les noms EXACTS — ne pas les modifier."
)


def load_df(path: Path) -> pd.DataFrame:
    sep_candidates = [";", ",", "\t", "|"]
    for sep in sep_candidates:
        for enc in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
                if len(df.columns) > 1:
                    print(f"  Chargé : {len(df):,} lignes, {len(df.columns)} colonnes [{enc}, sep='{sep}']")
                    return df
            except Exception:
                pass
    raise ValueError(f"Impossible de lire {path}")


def main() -> None:
    print(f"\n{'='*60}")
    print("Pipeline actuariel — exécution autonome")
    print(f"{'='*60}\n")

    # Charger les données
    if not DATA_FILE.exists():
        print(f"ERREUR : fichier non trouvé : {DATA_FILE}", file=sys.stderr)
        sys.exit(1)

    print(f"Données : {DATA_FILE.name}")
    df = load_df(DATA_FILE)

    # Importer l'agent
    sys.path.insert(0, str(ROOT))
    from agents.mortality.agents.graph import stream_agent

    history = [{"role": "user", "content": PROMPT}]
    data_store: dict = {}
    all_events: list[dict] = []

    print("\nDémarrage de l'agent...\n")

    step = 0
    for event in stream_agent(history, df=df, data_store=data_store, catalogue_level="full"):
        ev_type = event.get("type")
        step += 1

        # Log brut (sans les images base64 pour la lisibilité)
        log_entry = {k: v for k, v in event.items() if k != "tool_call_id"}
        if "result" in log_entry and isinstance(log_entry["result"], dict):
            log_entry["result"] = {
                k: ("<image_b64_tronquée>" if k == "image_b64" else
                    (f"[{len(v)} lignes]" if isinstance(v, list) and len(v) > 5 else v))
                for k, v in log_entry["result"].items()
            }
        all_events.append({"step": step, **log_entry})

        # Affichage console
        if ev_type == "tool_call":
            tool = event.get("tool", "")
            fn   = event.get("function_name", "")
            params = event.get("params", {})
            print(f"[{step:02d}] APPEL  → {tool}.{fn}")
            for k, v in params.items():
                print(f"       {k}: {v}")

        elif ev_type == "tool_result":
            fn     = event.get("function_name", "")
            result = event.get("result", {})
            has_err = "erreur" in result
            status = "ERREUR" if has_err else "OK"
            keys = [k for k in result if k not in ("erreur", "image_b64")]
            print(f"[{step:02d}] RÉSULT ← {fn} [{status}] — clés: {', '.join(keys)}")
            if has_err:
                print(f"       ⚠ {result['erreur']}")

        elif ev_type == "message":
            content = event.get("content", "")
            lines = content.strip().split("\n")
            prefix = f"[{step:02d}] MSG   "
            print(f"{prefix}  {lines[0][:120]}")
            for line in lines[1:5]:
                print(f"{'':10}  {line[:120]}")
            if len(lines) > 5:
                print(f"{'':10}  … ({len(lines)-5} lignes supplémentaires)")

        elif ev_type == "done":
            print(f"\n[{step:02d}] TERMINÉ")

        elif ev_type == "error":
            print(f"[{step:02d}] ERREUR : {event.get('message', '')}", file=sys.stderr)

    # Sauvegarder le log
    log_data = {
        "fichier_source": str(DATA_FILE),
        "modele": "gpt-4o",
        "nb_etapes": step,
        "events": all_events,
    }
    LOG_FILE.write_text(json.dumps(log_data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"\nLog complet sauvegardé : {LOG_FILE}")
    if Path(PDF_OUT).exists():
        print(f"PDF généré : {PDF_OUT}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
