"""
agents/report/pipeline/04_redaction.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPE 4 — Boucle Python déterministe + LLM par section (parallèle)

Reçoit le ReportPlan enrichi (étape 03).
Pour chaque section :
  1. Appelle les tools tableaux/graphiques (déterministe)
  2. Appelle GPT-4o avec le prompt de section enrichi (RAG inclus)
  3. Stocke le résultat dans section_outputs via write_section

Les sections sont traitées en parallèle via ThreadPoolExecutor :
  - Chaque thread reçoit un snapshot read-only du data_store
  - Pas d'écriture partagée pendant l'exécution parallèle
  - Les résultats sont écrits séquentiellement à la fin (ordre du plan préservé)
  - max_workers=5 pour éviter les 429 OpenAI TPM

Interface publique :
    redact_plan(plan, data_store) -> dict
        retourne data_store mis à jour avec section_outputs rempli
"""
from __future__ import annotations

import concurrent.futures
import logging
from typing import Any

log = logging.getLogger(__name__)

_MAX_TOKENS_NARRATIVE = 1200
_TEMPERATURE          = 0.4   # Faible : style professionnel, peu créatif


# ── Appels outils déterministes ───────────────────────────────────────────────

def _run_tables(section, data_store: dict) -> list[dict]:
    """
    Appelle table_renderer pour chaque spec tableau de la section.
    Retourne la liste des résultats (html + rows).
    Ne lève pas d'exception — erreurs loggées et ignorées.
    """
    results = []
    if not section.table_specs:
        return results

    try:
        from tools.build_pdf.table_renderer import render_table_from_spec
    except ImportError:
        log.warning("[04_redaction] table_renderer indisponible")
        return results

    context = {**data_store, **(section.context_snapshot or {})}

    for spec in section.table_specs:
        try:
            _, html, rows = render_table_from_spec(spec, context)
            if rows:
                results.append({"spec": spec, "html": html, "rows": rows})
                # Stocker pour write_section
                data_store["_last_table_rows"] = rows
                log.info("[04_redaction] tableau '%s' rendu (%d lignes)",
                         spec.get("id", "?"), len(rows))
        except Exception as exc:
            log.warning("[04_redaction] tableau '%s' échoué : %s", spec.get("id", "?"), exc)

    return results


def _run_stats(section, data_store: dict) -> list[dict]:
    """
    Appelle render_statistical_output pour chaque spec stat de la section.
    """
    results = []
    if not section.stat_specs:
        return results

    try:
        from tools.build_pdf.table_renderer import render_statistical_output
    except ImportError:
        log.warning("[04_redaction] render_statistical_output indisponible")
        return results

    context = {**data_store, **(section.context_snapshot or {})}

    for spec in section.stat_specs:
        try:
            _, html, rows = render_statistical_output(spec, context)
            if rows:
                results.append({"spec": spec, "html": html, "rows": rows})
                data_store["_last_table_rows"] = rows
                log.info("[04_redaction] stat '%s' rendu", spec.get("type", "?"))
        except Exception as exc:
            log.warning("[04_redaction] stat '%s' échoué : %s", spec.get("type", "?"), exc)

    return results


def _run_graphs(section, data_store: dict) -> list[str]:
    """
    Appelle graph_from_spec pour chaque spec graphique de la section.
    Retourne la liste des chemins PNG générés.
    """
    paths = []
    if not section.graph_specs:
        return paths

    try:
        from tools.graphs.graph_from_spec import generate_graph_from_spec
    except ImportError:
        log.warning("[04_redaction] graph_from_spec indisponible")
        return paths

    context = {**data_store, **(section.context_snapshot or {})}

    for spec in section.graph_specs:
        try:
            path = generate_graph_from_spec(spec, context)
            if path:
                paths.append(path)
                data_store["_last_graph_path"] = path
                log.info("[04_redaction] graphique '%s' généré : %s",
                         spec.get("id", "?"), path)
        except Exception as exc:
            log.warning("[04_redaction] graphique '%s' échoué : %s",
                        spec.get("id", "?"), exc)

    return paths


# ── Appel LLM de rédaction ────────────────────────────────────────────────────

def _build_redaction_prompt(section, table_results: list, graph_paths: list) -> str:
    """
    Finalise le prompt de rédaction en ajoutant :
    - les résultats des tableaux (HTML compact)
    - la liste des graphiques générés
    """
    prompt = section.prompt

    if table_results:
        prompt += "\n\n## Tableaux générés (résultats disponibles pour la rédaction)"
        for tr in table_results:
            name = tr["spec"].get("name", tr["spec"].get("id", "tableau"))
            rows = tr["rows"]
            # Résumé compact : en-tête + 3 premières lignes
            if len(rows) > 1:
                header = " | ".join(str(c) for c in rows[0])
                sample = "\n".join(
                    " | ".join(str(v) for v in row)
                    for row in rows[1:4]
                )
                prompt += f"\n\n**{name}**\n```\n{header}\n{sample}\n...\n```"

    if graph_paths:
        prompt += "\n\n## Graphiques générés"
        for p in graph_paths:
            prompt += f"\n- {p}"
        prompt += (
            "\nCes graphiques sont intégrés dans le rapport. "
            "Fais-y référence dans le texte (ex: 'La figure X montre...')."
        )

    prompt += (
        "\n\n## Consigne finale"
        "\nRédige maintenant le texte narratif de cette section."
        "\nStyle : professionnel, actuariel, en français."
        "\nNe répète pas les données brutes déjà dans les tableaux."
        "\nConclus la section par une phrase de synthèse."
    )

    return prompt


_SYSTEM_PROMPT_REDACTION = """\
Tu es un actuaire senior spécialisé dans la rédaction de rapports de certification de tables de mortalité.
Tu rédiges en français, style professionnel et précis.
Tu cites uniquement des chiffres présents dans les données fournies.
Tu ne calcules jamais une valeur manquante.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## CHARTE DE STYLE — À RESPECTER SCRUPULEUSEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 1. Structure du texte

Utilise exclusivement ce markup. Il sera rendu visuellement dans le rapport final :

  ## Titre de sous-section
      → Titre niveau 2 (bleu gras). Utilise pour chaque grande partie de ta section.
      → Exemple : ## Méthode de calcul des taux bruts

  ### Titre de paragraphe
      → Titre niveau 3 (bleu moyen). Utilise pour les sous-parties.
      → Exemple : ### Critères d'exclusion des données

  - item de liste
      → Liste à puces. Une ligne par item. N'imbrique pas.
      → Exemple :
        - Contrats sans exposition positive
        - Sinistres non classifiés décès

  > Note ou avertissement
      → Bloc en retrait, fond gris pâle. Pour les mises en garde et précisions techniques.
      → Exemple : > Les âges extrêmes (< 30 ans et > 85 ans) ont été exclus de l'analyse.

  Texte normal
      → Paragraphes justifiés. Sépare les paragraphes par une ligne vide.

  **texte en gras**
      → Emphase inline. Utilise pour les termes clés et les résultats chiffrés importants.
      → Exemple : Le SMR global est **0,748**, soit une mortalité d'expérience inférieure de 25%...

### 2. Formules mathématiques — OBLIGATOIRE

Toutes les expressions mathématiques DOIVENT être en notation LaTeX. Elles sont rendues
par le moteur LaTeX natif du rapport (qualité documentaire).

  $expression$     → formule inline dans une phrase
  $$expression$$   → formule en bloc, centrée sur sa propre ligne

Exemples CORRECTS :
  "Le taux brut est $q_x = D_x / E_x$ où $D_x$ désigne les décès observés."
  "Le SMR est défini par : $$\\text{SMR} = \\frac{\\sum_x D_x^{\\text{obs}}}{\\sum_x D_x^{\\text{att}}}$$"
  "L'intervalle de confiance bilatéral à $95\\%$ est $[\\hat{q}_x - 1{,}96\\,\\hat{\\sigma}_x;\\ \\hat{q}_x + 1{,}96\\,\\hat{\\sigma}_x]$."
  "Le lissage minimise : $$\\sum_x w_x(q_x - z_x)^2 + \\lambda\\sum_x(\\Delta^2 z_x)^2$$"
  "avec $\\lambda = 100$ le paramètre de lissage retenu."

Exemples INTERDITS (notation ASCII) :
  ✗ "q_x = D_x/E_x"          → écrire : "$q_x = D_x / E_x$"
  ✗ "SMR = 0.748"             → écrire : "$\\text{SMR} = 0{,}748$"
  ✗ "IC 95%"                  → écrire : "IC à $95\\%$"
  ✗ "lambda=100"              → écrire : "$\\lambda = 100$"

### 3. Conventions typographiques

  - Décimales : virgule française (0,748 et non 0.748)
  - Milliers : espace fine (346 600 et non 346600) — dans le texte courant seulement
  - Pourcentages : toujours collés au chiffre avec le signe % (25 %)
  - Guillemets : « guillemets français »

### 4. Structure type d'une section

  ## Contexte et objectifs
  [1 paragraphe d'introduction]

  ## [Sous-section 1 : méthode / données / résultats]
  [2-3 paragraphes]
  [liste si nécessaire]

  ## [Sous-section 2 : analyse / interprétation]
  [2-3 paragraphes]
  [note si mise en garde]

  ## Synthèse
  [1 paragraphe de conclusion de la section]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def _call_llm_redaction(prompt: str) -> str:
    """
    Appelle GPT-4o pour rédiger le texte narratif de la section.
    Retourne le texte rédigé, ou "" en cas d'échec.
    """
    try:
        import openai
        from agents.mortality.agents._utils import call_with_retry

        client = openai.OpenAI()
        response = call_with_retry(
            client,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT_REDACTION},
                {"role": "user",   "content": prompt},
            ],
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS_NARRATIVE,
        )
        return (response.choices[0].message.content or "").strip()

    except Exception as exc:
        log.error("[04_redaction] LLM rédaction échoué : %s", exc)
        return ""


# ── Stockage dans section_outputs ─────────────────────────────────────────────

def _write_section(section_id: str, text: str, data_store: dict,
                   table_caption: str = "", graph_caption: str = "") -> None:
    """
    Appelle write_section pour accumuler texte + tableau + graphique
    dans data_store["section_outputs"][section_id].
    """
    try:
        from tools.build_pdf.write_section import run as _ws_run
        _ws_run(
            data=data_store,
            params={
                "section_id":    section_id,
                "text":          text,
                "table_caption": table_caption,
                "graph_caption": graph_caption,
                "status":        "done" if text else "partial",
            },
        )
    except Exception as exc:
        log.error("[04_redaction] write_section '%s' échoué : %s", section_id, exc)


# ── Traitement parallèle d'une section (thread-safe) ─────────────────────────

def _process_section_parallel(sec, ds_snapshot: dict) -> tuple[str, dict]:
    """
    Traite une section dans un thread séparé.
    Thread-safe : reçoit un snapshot read-only du data_store,
    n'écrit rien dans le data_store partagé.

    Retourne (section_id, result_dict).
    """
    if not sec.ready:
        return sec.section_id, {
            "text": "", "table_caption": "", "graph_caption": "",
            "status": "skipped", "n_tables": 0, "n_graphs": 0,
        }

    # Copie locale pour isoler les écritures (_last_table_rows, _last_graph_path)
    local_ds = dict(ds_snapshot)

    table_results = _run_tables(sec, local_ds)
    stat_results  = _run_stats(sec, local_ds)
    graph_paths   = _run_graphs(sec, local_ds)
    all_tables    = table_results + stat_results

    prompt = _build_redaction_prompt(sec, all_tables, graph_paths)
    text   = _call_llm_redaction(prompt)

    return sec.section_id, {
        "text":          text,
        "table_caption": all_tables[-1]["spec"].get("name", "") if all_tables else "",
        "graph_caption": sec.graph_specs[-1].get("name", "") if sec.graph_specs else "",
        "status":        "done" if text else "partial",
        "n_tables":      len(all_tables),
        "n_graphs":      len(graph_paths),
    }


# ── Point d'entrée public ─────────────────────────────────────────────────────

def redact_plan(plan, data_store: dict) -> dict:
    """
    Traite toutes les sections du ReportPlan enrichi en parallèle.

    Architecture thread-safe :
      - Snapshot du data_store passé en lecture seule à chaque worker
      - max_workers=5 pour éviter les 429 OpenAI TPM
      - Écriture séquentielle finale (ordre du plan préservé)

    Args:
        plan       : ReportPlan enrichi par 03_completion_plan
        data_store : résultats du BuilderAgent (modifié en place)

    Returns:
        data_store mis à jour avec section_outputs rempli
    """
    section_outputs = data_store.setdefault("section_outputs", {})

    # Séparer sections prêtes / skippées
    ready   = [sec for sec in plan.sections if sec.ready]
    skipped = [sec for sec in plan.sections if not sec.ready]

    # Sections non prêtes → skip immédiat, pas de thread
    for sec in skipped:
        section_outputs[sec.section_id] = {
            "text": "", "tables": [], "table_captions": [],
            "graphs": [], "graph_captions": [], "status": "skipped",
        }
        log.info("[04_redaction] '%s' — skipped (données manquantes : %s)",
                 sec.section_id, sec.missing_inputs)

    if not ready:
        log.info("[04_redaction] aucune section prête — terminé")
        return data_store

    # Snapshot read-only pour les workers (évite les conflits d'écriture)
    ds_snapshot = dict(data_store)

    # Limiter le parallélisme : 5 appels LLM simultanés max (429 protection)
    max_workers = min(len(ready), 5)
    log.info("[04_redaction] %d sections en parallèle (max_workers=%d)",
             len(ready), max_workers)

    results: dict[str, dict] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_section_parallel, sec, ds_snapshot): sec.section_id
            for sec in ready
        }
        for future in concurrent.futures.as_completed(futures):
            sid = futures[future]
            try:
                section_id, result = future.result()
                results[section_id] = result
                log.info("[04_redaction] '%s' terminée (%d chars, %d tableaux, %d graphiques)",
                         section_id, len(result["text"]),
                         result["n_tables"], result["n_graphs"])
            except Exception as exc:
                log.error("[04_redaction] '%s' — exception : %s", sid, exc)
                results[sid] = {
                    "text": "", "table_caption": "", "graph_caption": "",
                    "status": "error", "n_tables": 0, "n_graphs": 0,
                }

    # Écriture séquentielle pour préserver l'ordre du plan et éviter les conflits
    n_done = 0
    for sec in ready:
        r = results.get(sec.section_id, {})
        _write_section(
            section_id    = sec.section_id,
            text          = r.get("text", ""),
            data_store    = data_store,
            table_caption = r.get("table_caption", ""),
            graph_caption = r.get("graph_caption", ""),
        )
        n_done += 1

    log.info("[04_redaction] terminé — %d sections rédigées, %d skippées",
             n_done, len(skipped))
    return data_store
