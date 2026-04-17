"""
TOOL CONTRACT — build_pdf.session_log
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : build_pdf.session_log
domain        : descriptive
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Génère un fichier TXT de log de la session actuarielle. Contient le
raisonnement de l'agent, la séquence complète d'appels avec paramètres
et résultats, et un bloc JSON "REPLAY" permettant de rejouer identiquement
la même analyse lors d'une session future.

WHEN TO USE
-----------
Proposer systématiquement à la fin d'une analyse complète. Particulièrement
utile pour les clients souhaitant rejouer l'analyse sur un nouveau portefeuille
ou valider la reproductibilité des résultats.

WHEN NOT TO USE
---------------
Ne pas appeler si _call_log est vide (aucun appel de tool effectué).

PREREQUISITES
-------------
required_tools: [any tools called during the session]
required_data_store_keys:
  - _call_log (requis — liste des appels de session)

INPUTS
------
params:
  output_path:
    type    : string
    values  : chemin de fichier .txt
    default : /tmp/session_actuarielle.txt
    note    : L'interface gère le téléchargement. Ne pas exposer au client.
  portfolio_info:
    type    : string
    values  : texte court
    default : ""
    note    : Description du portefeuille (ex: "45 000 lignes, 8 colonnes, 2010-2023").

OUTPUTS
-------
data_store_keys_written: []
return_payload:
  succes      : bool
  output_path : str
  nb_steps    : int — nombre d'étapes enregistrées dans le log

QUALITY GATES
-------------
BLOCKING:
  - _call_log vide → log généré sans étapes. Informer le client.
NON-BLOCKING: []

ERROR HANDLING
--------------
error: "[exception lors de l'écriture du fichier]"
  → cause  : Erreur système lors de l'écriture.
  → action : Vérifier les droits d'accès au répertoire /tmp/.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Proposer à la fin d'une analyse : "Souhaitez-vous un log de session
  pour rejouer cette analyse ?" Le bloc JSON REPLAY dans le log permettra
  à l'agent de reproduire exactement la même séquence.
  Ne jamais mentionner le chemin output_path dans la réponse au client.
exemplar_query: >
  Comment permettre au client de rejouer une analyse actuarielle identique ?

CATALOGUE METADATA
------------------
display_name      : Log de session (rejouer l'analyse)
short_description : Génère un fichier TXT avec le log complet et un bloc JSON de replay.
domain            : descriptive
capability_group  : reporting
depends_on        : []
required_by       : []
client_visible    : true
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def run(data: dict | None, params: dict | None = None) -> dict:
    data   = data   or {}
    params = params or {}

    output_path    = params.get("output_path", "/tmp/session_actuarielle.txt")
    portfolio_info = params.get("portfolio_info", "")

    call_log      = data.get("_call_log", [])
    reasoning_log = data.get("_reasoning_log", [])

    sep = "=" * 70
    lines = []

    # ── En-tête ───────────────────────────────────────────────────────────────
    lines += [
        sep,
        "SESSION LOG — Agent Actuariat v2.0",
        sep,
        f"Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if portfolio_info:
        lines.append(f"Données: {portfolio_info}")
    lines.append("")

    # ── Raisonnement ──────────────────────────────────────────────────────────
    if reasoning_log:
        lines += [sep, "RAISONNEMENT DE L'AGENT", sep]
        for i, text in enumerate(reasoning_log, 1):
            lines.append(f"\n[Message {i}]")
            lines.append(text)
        lines.append("")

    # ── Séquence d'appels ─────────────────────────────────────────────────────
    if call_log:
        lines += [sep, "SÉQUENCE D'APPELS DE FONCTIONS", sep]
        for entry in call_log:
            step        = entry.get("step", "?")
            tool        = entry.get("tool", "")
            fn          = entry.get("function_name", "")
            params_used = entry.get("params", {})
            summary     = entry.get("result_summary", {})
            has_error   = entry.get("has_error", False)

            status = "ERREUR" if has_error else "OK"
            lines.append(f"\nSTEP {step} — {tool}.{fn}  [{status}]")
            lines.append(f"  Paramètres : {json.dumps(params_used, ensure_ascii=False)}")
            for k, v in summary.items():
                if k not in ("traceback",):
                    lines.append(f"  {k} : {v}")
        lines.append("")

    # ── Bloc REPLAY (machine-readable) ────────────────────────────────────────
    lines += [sep, "REPLAY — Bloc JSON (ne pas modifier)", sep]
    replay = {
        "version":        "2.0",
        "portfolio_info": portfolio_info,
        "steps": [
            {
                "step":          e.get("step"),
                "tool":          e.get("tool"),
                "function_name": e.get("function_name"),
                "params":        e.get("params", {}),
            }
            for e in call_log
            if not e.get("has_error")
        ],
    }
    lines.append(json.dumps(replay, ensure_ascii=False, indent=2))
    lines += [
        "",
        sep,
        "Pour rejouer : uploadez votre fichier CSV et envoyez ce fichier à l'agent.",
        "L'agent reproduira la même séquence avec les mêmes paramètres.",
        sep,
    ]

    content = "\n".join(lines)
    try:
        Path(output_path).write_text(content, encoding="utf-8")
        return {
            "succes":      True,
            "output_path": output_path,
            "nb_steps":    len(call_log),
        }
    except Exception as exc:
        return {"erreur": str(exc), "succes": False}
