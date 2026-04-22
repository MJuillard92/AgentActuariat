"""
agents/master/disambiguation.py
Désambiguation d'intention + vérification des prérequis avant routing.

Flow :
  1. classify_intent()      — règles rapides, LLM si ambigu
  2. load_task_prerequisites()  — lit catalogue.yaml → user_inputs_required
  3. check_prerequisites()  — vérifie ce qui manque dans state
  4. suggest_column_mapping() — LLM sur les colonnes du DataFrame

Chaque fonction retourne un dict structuré consommé par master_node.
"""
from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

if TYPE_CHECKING:
    pass

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CATALOGUE_PATH = _PROJECT_ROOT / "tools" / "catalogue.yaml"

# ── Mots-clés d'intention ─────────────────────────────────────────────────────

_INTENT_RULES: dict[str, list[str]] = {
    "mortality_table": [
        "loi de mortalité", "table de mortalité", "table d'expérience",
        "calcule", "lance le pipeline", "exposition", "taux bruts", "lissage",
        "builder", "smr", "benchmarking", "validation", "construction",
        "pipeline actuariel", "certification",
        # expressions plus larges
        "étude de mortalité", "étude actuarielle", "mortalité", "mortalite",
        "lance l'étude", "lance l'analyse", "lance une étude", "lance",
        "construis", "construire", "démarrer", "démarre", "demarrer", "demarre",
        "analyse les données", "analyse de survie", "qx", "q_x",
    ],
    "report": [
        "rapport", "génère le rapport", "rédige", "pdf", "rédaction",
        "write", "rapport de certification", "go_write",
    ],
    "descriptive": [
        "analyse descriptive", "résumé", "résume", "segmentation",
        "distribution", "pyramide des âges", "séries temporelles",
        "qualité des données", "exploration",
    ],
    "replay": [
        "rejoue", "replay", "relance", "reprends la session",
        "restaure", "même analyse",
    ],
}

# Colonnes actuarielles attendues — clé = nom canonique, valeur = alias possibles
EXPECTED_COLUMNS: dict[str, list[str]] = {
    "date_naissance": [
        "dob", "birth_date", "naissance", "date_naiss", "birthdate",
        "birth", "dat_nais", "date_de_naissance", "dn",
    ],
    "date_entree": [
        "entry_date", "debut", "date_debut", "souscription", "start_date",
        "entry", "date_entree", "dat_entree", "debut_contrat", "de",
    ],
    "date_sortie": [
        "exit_date", "fin", "date_fin", "resiliation", "end_date",
        "exit", "date_sortie", "dat_sortie", "fin_contrat", "ds",
    ],
    "cause_sortie": [
        "cause", "motif", "exit_reason", "statut", "status", "reason",
        "cause_sortie", "etat", "type_sortie", "cs",
    ],
    "sexe": [
        "gender", "sex", "genre", "sex_code", "sexe", "s", "g",
    ],
}


# ── Chargement du catalogue ───────────────────────────────────────────────────

_catalogue_cache: dict | None = None


def _load_catalogue() -> dict:
    global _catalogue_cache
    if _catalogue_cache is None:
        try:
            with open(_CATALOGUE_PATH, encoding="utf-8") as f:
                _catalogue_cache = yaml.safe_load(f) or {}
        except Exception:
            _catalogue_cache = {}
    return _catalogue_cache


def load_task_prerequisites(task_type: str) -> list[dict]:
    """
    Lit le catalogue.yaml et retourne la liste des user_inputs_required
    pour les tools du task_type donné.

    Retourne une liste de dicts :
      {key, source, label, type, required, options?, placeholder?, description}
    """
    catalogue = _load_catalogue()
    tools_section = catalogue.get("tools") or {}

    # Mapping task_type → tools pertinents
    _TASK_TOOLS = {
        "mortality_table": ["builder.exposure", "builder.benchmarking"],
        "report":          [],
        "descriptive":     [],
        "replay":          [],
    }
    relevant_tools = _TASK_TOOLS.get(task_type, [])

    prereqs: list[dict] = []
    seen_keys: set[str] = set()

    for tool_name in relevant_tools:
        tool_def = tools_section.get(tool_name) or {}
        for inp in tool_def.get("user_inputs_required") or []:
            key = inp.get("key", "")
            if key and key not in seen_keys:
                seen_keys.add(key)
                prereqs.append(inp)

    return prereqs


# ── Classification d'intention ────────────────────────────────────────────────

def classify_intent(
    message: str,
    data_store: dict | None = None,
) -> dict:
    """
    Classifie l'intention depuis le message + contexte.

    Retourne :
      {task_type, confidence, method}
      method = "rules" | "llm" | "context"
    """
    data_store = data_store or {}
    text = message.lower()

    # Contexte : si des calculs existent déjà et que le message parle de rapport
    if data_store.get("smoothed_table"):
        for kw in _INTENT_RULES["report"]:
            if kw in text:
                return {"task_type": "report", "confidence": 0.95, "method": "context"}

    # Règles rapides
    scores: dict[str, int] = {}
    for intent, keywords in _INTENT_RULES.items():
        score = sum(1 for kw in keywords if kw in text)
        if score:
            scores[intent] = score

    if scores:
        best = max(scores, key=lambda k: scores[k])
        total = sum(scores.values())
        confidence = scores[best] / total if total else 0.5
        if confidence >= 0.5:
            return {"task_type": best, "confidence": confidence, "method": "rules"}

    # Ambiguïté → LLM classification
    llm_result = _classify_with_llm(message, data_store)
    if llm_result:
        return llm_result

    # Fallback : on suppose une demande de table de mortalité (usage principal de l'outil)
    return {"task_type": "mortality_table", "confidence": 0.3, "method": "fallback"}


def _classify_with_llm(message: str, data_store: dict) -> dict | None:
    """Appelle GPT-4o pour classer l'intention si les règles ne suffisent pas."""
    try:
        import openai
        from agents.mortality.agents._utils import call_with_retry

        ds_context = []
        if data_store.get("exposure_table"):
            ds_context.append("exposition calculée")
        if data_store.get("smoothed_table"):
            ds_context.append("table lissée disponible")

        prompt = (
            "Classifie l'intention dans ce message d'un utilisateur d'un outil actuariel.\n"
            "Réponds en JSON uniquement avec : {\"task_type\": \"...\", \"confidence\": 0.0-1.0}\n\n"
            f"Types possibles : mortality_table, report, descriptive, replay, unknown\n\n"
            f"Contexte session : {', '.join(ds_context) or 'aucun calcul effectué'}\n\n"
            f"Message : {message}\n\nJSON :"
        )
        client = openai.OpenAI()
        response = call_with_retry(
            client,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=80,
        )
        import json
        result = json.loads(response.choices[0].message.content or "{}")
        task_type = result.get("task_type", "unknown")
        confidence = float(result.get("confidence", 0.5))
        return {"task_type": task_type, "confidence": confidence, "method": "llm"}
    except Exception:
        return None


# ── Vérification des prérequis ────────────────────────────────────────────────

def check_prerequisites(
    task_type: str,
    df_json: str | None,
    data_store: dict,
) -> dict:
    """
    Vérifie que tous les prérequis du task_type sont disponibles dans l'état.

    Retourne :
      {
        ready: bool,
        missing: list[dict],   — prérequis manquants (même format que user_inputs_required)
        needs_column_mapping: bool,
        needs_form: bool,
      }
    """
    prereqs = load_task_prerequisites(task_type)
    study_plan = data_store.get("study_plan") or {}
    column_mapping = data_store.get("column_mapping") or {}
    column_mapping_confirmed = data_store.get("column_mapping_confirmed", False)

    missing: list[dict] = []

    for inp in prereqs:
        key = inp.get("key", "")
        source = inp.get("source", "")
        required = inp.get("required", True)

        if not required:
            continue

        if source == "data_store":
            # Cas spécial : mapping de colonnes
            if key == "column_mapping":
                if not df_json:
                    missing.append({**inp, "_reason": "Aucun fichier CSV chargé."})
                elif not column_mapping and not column_mapping_confirmed:
                    # Pas encore de mapping du tout
                    missing.append({**inp, "_reason": "Mapping non confirmé."})
                elif column_mapping and not column_mapping_confirmed:
                    # Mapping partiel : des colonnes requises manquent encore
                    unmatched = data_store.get("column_mapping_unmatched") or []
                    required_roles = {"date_naissance", "date_entree", "date_sortie", "cause_sortie"}
                    still_missing = [r for r in unmatched if r in required_roles]
                    if still_missing:
                        inp_copy = {**inp, "_reason": f"Colonnes manquantes : {still_missing}"}
                        missing.append(inp_copy)
                # Si column_mapping_confirmed = True → pas de missing, on ne pose pas la question
            else:
                val = data_store.get(key)
                if val is None or val == [] or val == {}:
                    missing.append({**inp, "_reason": f"Clé '{key}' absente du data_store."})

        elif source == "study_plan":
            val = study_plan.get(key)
            if val is None or val == "" or val == [] or val == {}:
                missing.append({**inp, "_reason": f"Paramètre '{key}' non défini."})

    needs_col_map = any(m.get("type") == "column_mapping" for m in missing)
    needs_form = any(m.get("type") != "column_mapping" for m in missing)

    return {
        "ready":                not bool(missing),
        "missing":              missing,
        "needs_column_mapping": needs_col_map,
        "needs_form":           needs_form,
    }


# ── Suggestion de mapping de colonnes ────────────────────────────────────────

def suggest_column_mapping(df_columns: list[str]) -> dict[str, str | None]:
    """
    Propose un mapping initial colonnes CSV → champs actuariels.

    Stratégie :
      1. Correspondance exacte (insensible à la casse)
      2. Correspondance par alias
      3. LLM si toujours ambigu

    Retourne {canonical_name: csv_column_name | None}
    """
    cols_lower = {c.lower(): c for c in df_columns}
    mapping: dict[str, str | None] = {}

    for canonical, aliases in EXPECTED_COLUMNS.items():
        found = None
        # Correspondance exacte
        if canonical in cols_lower:
            found = cols_lower[canonical]
        else:
            # Correspondance par alias
            for alias in aliases:
                if alias in cols_lower:
                    found = cols_lower[alias]
                    break
        mapping[canonical] = found

    # Si des colonnes restent non mappées → LLM
    unmapped = [k for k, v in mapping.items() if v is None]
    if unmapped:
        llm_suggestions = _suggest_mapping_with_llm(df_columns, unmapped)
        for k, v in llm_suggestions.items():
            if mapping.get(k) is None:
                mapping[k] = v

    return mapping


def _suggest_mapping_with_llm(
    df_columns: list[str],
    unmapped_canonical: list[str],
) -> dict[str, str | None]:
    """LLM pour résoudre les colonnes non trouvées par règles."""
    try:
        import openai
        import json
        from agents.mortality.agents._utils import call_with_retry

        descriptions = {
            "date_naissance": "date de naissance des assurés",
            "date_entree":    "date d'entrée en observation (souscription/adhésion)",
            "date_sortie":    "date de sortie d'observation (résiliation/fin de contrat)",
            "cause_sortie":   "cause de sortie (décès=1, autre=0 ou similaire)",
            "sexe":           "sexe des assurés (H/F ou 0/1 ou M/F)",
        }

        targets_desc = "\n".join(
            f"  - {k}: {descriptions.get(k, k)}"
            for k in unmapped_canonical
        )
        prompt = (
            f"Colonnes disponibles dans le CSV : {df_columns}\n\n"
            f"Trouve la colonne qui correspond le mieux à chaque champ :\n{targets_desc}\n\n"
            "Réponds en JSON : {\"date_naissance\": \"nom_colonne_ou_null\", ...}\n"
            "Si aucune colonne ne correspond, mettre null.\n\nJSON :"
        )
        client = openai.OpenAI()
        response = call_with_retry(
            client,
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        return json.loads(response.choices[0].message.content or "{}")
    except Exception:
        return {k: None for k in unmapped_canonical}


# ── Détection du stage value_mapping (US-13) ─────────────────────────────────

def detect_value_mapping_stage(records, enum_specs: dict) -> dict:
    """Inspecte les colonnes enum des records, propose un mapping.

    Retourne un dict avec `stage` :
      - "skip"                 : toutes les valeurs sont déjà conformes
      - "needs_value_mapping"  : des valeurs non conformes mappables → suggestion
      - "blocked"              : des valeurs n'ont pas de mapping évident → message
    """
    from tools.master.suggest_value_mapping import run as _suggest

    out = _suggest({"records": records, "enum_specs": enum_specs}, {})
    suggestion = {k: v for k, v in (out["value_mapping"] or {}).items() if v}
    unmapped = {k: v for k, v in (out["unmapped"] or {}).items() if v}

    if unmapped:
        parts = [f"{col}: {vals}" for col, vals in unmapped.items()]
        message = (
            "Certaines valeurs de votre fichier ne peuvent pas être mises en "
            "correspondance automatiquement : "
            + " ; ".join(parts)
            + ". Corrigez votre fichier ou précisez la correspondance."
        )
        return {"stage": "blocked", "suggestion": suggestion,
                "unmapped": unmapped, "message": message}

    if suggestion:
        return {"stage": "needs_value_mapping", "suggestion": suggestion,
                "unmapped": {}}

    return {"stage": "skip", "suggestion": {}, "unmapped": {}}


# ── Normalisation des records (US-14) ────────────────────────────────────────

def maybe_normalize_records(data_store: dict, df_json: str | None) -> dict | None:
    """Retourne les updates data_store à appliquer si normalisation due.

    Déclenche uniquement quand :
      column_mapping_confirmed ET value_mapping_confirmed ET NOT records_normalized.
    Renvoie None sinon (no-op).
    """
    if not data_store.get("column_mapping_confirmed"):
        return None
    if not data_store.get("value_mapping_confirmed"):
        return None
    if data_store.get("records_normalized"):
        return None
    if not df_json:
        return None

    import pandas as pd
    from tools.master.normalize_records import run as _normalize

    df_in = pd.read_json(StringIO(df_json), orient="split")
    # data_store stocke column_mapping en format {canonical: csv_col} (cf.
    # build_mapping_report). Le tool normalize_records attend {old: new},
    # donc {csv_col: canonical} — on inverse.
    column_mapping_canonical = data_store.get("column_mapping") or {}
    column_mapping_for_tool = {v: k for k, v in column_mapping_canonical.items() if v}
    value_mapping = data_store.get("value_mapping") or {}

    result = _normalize(
        {"records": df_in, "column_mapping": column_mapping_for_tool, "value_mapping": value_mapping},
        {},
    )
    df_out = result["normalized_records"]

    audit_entry = {
        "column_mapping": dict(column_mapping_canonical),
        "value_mapping":  dict(value_mapping),
        "rows_in":        len(df_in),
        "rows_out":       len(df_out),
    }
    existing_audit = dict(data_store.get("_audit") or {})
    existing_audit["normalization"] = audit_entry

    return {
        "input_records":       df_out,
        "records_normalized":  True,
        "_audit":              existing_audit,
    }


# ── Point d'entrée principal ─────────────────────────────────────────────────

def run_disambiguation(
    message: str,
    df_json_or_ref: str | None,
    data_store: dict,
) -> dict:
    """
    df_json_or_ref : soit un JSON orient=split (legacy), soit un session_id
    (dataset_ref) depuis lequel le DataFrame est chargé via DatasetStore.
    """
    """
    Fonction principale appelée par master_node.

    Retourne un dict avec :
      status : "ready" | "needs_input" | "unclear"
      task_type : str
      confidence : float
      missing : list[dict]
      needs_column_mapping : bool
      needs_form : bool
      column_mapping_suggestion : dict  (si needs_column_mapping)
      form_fields : list[dict]          (si needs_form)
    """
    # 1. Classification d'intention
    intent = classify_intent(message, data_store)
    task_type  = intent["task_type"]
    confidence = intent["confidence"]

    if task_type == "unknown" or confidence < 0.3:
        return {
            "status":    "unclear",
            "task_type": task_type,
            "confidence": confidence,
            "message":   "Je n'ai pas bien compris votre demande. Pouvez-vous préciser ?",
        }

    # Résoudre df_json_or_ref → df_json (pour compatibilité avec check_prerequisites)
    # Accepte soit un JSON orient=split, soit un session_id (dataset_ref)
    df_json: str | None = None
    if df_json_or_ref:
        if df_json_or_ref.startswith("{") or df_json_or_ref.startswith("["):
            df_json = df_json_or_ref   # JSON direct (legacy)
        else:
            # C'est un session_id → charger le Parquet et sérialiser
            try:
                from session.dataset_store import DatasetStore
                df_loaded = DatasetStore.load_by_session(df_json_or_ref)
                if df_loaded is not None:
                    df_json = df_loaded.to_json(orient="split")
            except Exception:
                pass

    # 2bis. Étape value_mapping (US-13) : si column_mapping est confirmé
    # mais pas value_mapping, détecter les valeurs non conformes avant tout.
    if (
        data_store.get("column_mapping_confirmed")
        and not data_store.get("value_mapping_confirmed")
        and df_json
    ):
        try:
            import pandas as pd
            from knowledge_base.report_template.template_loader import load_enum_specs

            df = pd.read_json(StringIO(df_json), orient="split")
            col_map = data_store.get("column_mapping") or {}
            # Renommer vers les noms canoniques avant inspection des valeurs
            if col_map:
                reverse = {v: k for k, v in col_map.items() if v}
                df = df.rename(columns=reverse)
            enum_specs = load_enum_specs()
        except Exception:
            enum_specs = {}
            df = None

        if df is not None and enum_specs:
            vm = detect_value_mapping_stage(df, enum_specs)
            if vm["stage"] == "blocked":
                return {
                    "status":    "unclear",
                    "task_type": task_type,
                    "confidence": confidence,
                    "message":   vm["message"],
                }
            if vm["stage"] == "needs_value_mapping":
                return {
                    "status":                  "needs_input",
                    "task_type":               task_type,
                    "confidence":              confidence,
                    "needs_value_mapping":     True,
                    "needs_column_mapping":    False,
                    "needs_form":              False,
                    "value_mapping_suggestion": vm["suggestion"],
                    "form_fields":             [],
                }
            # stage == "skip" : rien à mapper, on marque comme confirmé pour
            # débloquer maybe_normalize_records en aval (sinon input_records
            # n'est jamais stocké → branche déterministe du Builder skippée).
            data_store["value_mapping_confirmed"] = True

    # 2. Vérification des prérequis
    prereq_check = check_prerequisites(task_type, df_json, data_store)

    if prereq_check["ready"]:
        return {
            "status":    "ready",
            "task_type": task_type,
            "confidence": confidence,
        }

    # 3. Construire la suggestion de mapping si nécessaire
    # Priorité : mapping déjà calculé par build_mapping_report au chargement du CSV
    col_suggestion = {}
    if prereq_check["needs_column_mapping"]:
        existing_mapping = data_store.get("column_mapping") or {}
        if existing_mapping:
            # Utiliser le mapping déjà détecté — ne pas rappeler le LLM
            col_suggestion = {k: existing_mapping.get(k) for k in EXPECTED_COLUMNS}
        elif df_json:
            try:
                import pandas as pd
                df = pd.read_json(StringIO(df_json), orient="split")
                col_suggestion = suggest_column_mapping(list(df.columns))
            except Exception:
                col_suggestion = {k: None for k in EXPECTED_COLUMNS}

    # Séparer champs formulaire (non mapping)
    form_fields = [
        m for m in prereq_check["missing"]
        if m.get("type") != "column_mapping"
    ]

    return {
        "status":                  "needs_input",
        "task_type":               task_type,
        "confidence":              confidence,
        "missing":                 prereq_check["missing"],
        "needs_column_mapping":    prereq_check["needs_column_mapping"],
        "needs_form":              prereq_check["needs_form"],
        "column_mapping_suggestion": col_suggestion,
        "df_columns":              _get_df_columns(df_json),   # df_json déjà résolu ci-dessus
        "form_fields":             form_fields,
    }


def _get_df_columns(df_json: str | None) -> list[str]:
    if not df_json:
        return []
    try:
        import pandas as pd
        df = pd.read_json(StringIO(df_json), orient="split")
        return list(df.columns)
    except Exception:
        return []
