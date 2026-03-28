"""
agent.py
Agent actuariel ReAct — boucle outil execute_python via l'API OpenAI.

Implémente le patron ReAct (Reasoning + Acting) :
  1. L'agent (LLM) reçoit un message utilisateur et un contexte (notebooks).
  2. Il réfléchit et décide d'appeler un outil (execute_python ou search_documentation).
  3. Le résultat de l'outil est renvoyé au LLM comme message "tool".
  4. La boucle reprend jusqu'à ce que le LLM réponde avec finish_reason="stop"
     (réponse finale) ou que MAX_ITERATIONS soit atteint (garde-fou).

Deux outils sont exposés au LLM :
  - execute_python        : exécute du code dans le kernel partagé (résultats persistants)
  - search_documentation  : recherche dans la base documentaire actuarielle (RAG externe)

Le kernel partagé est initialisé par make_kernel() dans workflow_executor.py et
contient tous les modules actuariels + les paramètres PARAMS. L'agent n'a pas
besoin de réimporter quoi que ce soit entre ses appels.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Generator

from dotenv import load_dotenv
from openai import OpenAI

import config

# ─────────────────────────────────────────────────────────────────────────────
# Planning : prompt système et fonction de planification
# ─────────────────────────────────────────────────────────────────────────────

PLANNING_SYSTEM_PROMPT = """\
Tu es un expert actuariel senior chargé de planifier une analyse quantitative.
À partir du message utilisateur, des données disponibles et de la doctrine actuarielle,
génère un plan d'analyse structuré et détaillé en macro-étapes.

RÈGLES :
- Entre 4 et 8 étapes maximum.
- Chaque étape doit nommer la MÉTHODE STATISTIQUE exacte utilisée (ex : estimateur de Kaplan-Meier,
  méthode de Whittaker-Henderson, Chain-Ladder, Bornhuetter-Ferguson, etc.).
- Inclure la FORMULE mathématique clé de chaque étape (notation actuarielle standard).
- Mentionner les ALTERNATIVES si plusieurs méthodes existent (ex : "Kaplan-Meier ou taux centraux").
- Si une adaptation des données est nécessaire (ex : table unisexe, recodage), signaler une étape
  "Code custom" avec justification.
- Ne génère PAS de code Python, seulement le plan méthodologique.
- Pour une table de mortalité TD/TF, les étapes typiques sont :
  1. Chargement / nettoyage (contrôle qualité, dates incohérentes, âges hors plage)
  2. Calcul des expositions (Kaplan-Meier central / initial, E_x = Σ durées_obs_ans)
  3. Taux bruts (q̂_x = D_x / E_x ou estimateur K-M : q̂_x = 1 - ∏(1 - d_i/n_i))
  4. Lissage (Whittaker-Henderson : min Σ(w_x(q̂_x-q_x)²) + λΣ(Δ²q_x)²)
  5. Validation statistique (test χ², SMR, intervalles de confiance Poisson)
  6. Positionnement réglementaire (abattements vs TH0002/TF0002, SMR global)

Réponds UNIQUEMENT avec un JSON valide (sans markdown autour) :
{
  "steps": [
    {
      "id": 1,
      "titre": "...",
      "description": "Description précise de ce que produit cette étape.",
      "methode": "Nom de la méthode statistique exacte utilisée",
      "formule": "Notation mathématique clé (ex: q̂_x = D_x / E_x)",
      "alternatives": "Méthodes alternatives si applicables (ou null)",
      "outils": ["module.fonction"],
      "custom_code": false,
      "obligatoire": true
    }
  ]
}
"""

_PLAN_FALLBACK: list[dict] = [
    {
        "id": 1,
        "titre": "Chargement et nettoyage",
        "description": "Charger le CSV et nettoyer les données.",
        "outils": ["data_prep.load_data", "data_prep.clean_data"],
        "obligatoire": True,
    },
    {
        "id": 2,
        "titre": "Calcul des expositions",
        "description": "Calculer l'exposition au risque par âge.",
        "outils": ["exposure.compute_exposure_by_age"],
        "obligatoire": True,
    },
    {
        "id": 3,
        "titre": "Taux bruts",
        "description": "Calculer les taux de mortalité bruts.",
        "outils": ["crude_rates.compute_crude_rates"],
        "obligatoire": False,
    },
    {
        "id": 4,
        "titre": "Lissage et validation",
        "description": "Lisser les taux et valider statistiquement.",
        "outils": ["smoothing.smooth_rates", "diagnostics.run_diagnostics"],
        "obligatoire": False,
    },
]


def plan_agent(
    user_message: str,
    kb_context: str,
    data_context: str,
    model: str | None = None,
) -> list[dict]:
    """Appelle l'API OpenAI pour générer un plan d'analyse actuariel structuré.

    Args:
        user_message:  Message de l'utilisateur décrivant l'analyse souhaitée.
        kb_context:    Extrait de la base de connaissances actuarielle (doctrine).
        data_context:  Informations sur les données disponibles (chemin CSV, sexe, domaine).
        model:         Modèle OpenAI à utiliser. None → utilise config.REASONING_MODEL.

    Returns:
        Liste de dicts représentant les étapes du plan.
        Retourne _PLAN_FALLBACK en cas d'échec.
    """
    _model = model or config.PLANNING_MODEL

    prompt_user = (
        f"Message utilisateur :\n{user_message}\n\n"
        f"Données disponibles :\n{data_context}\n\n"
        f"Doctrine actuarielle (extrait) :\n{kb_context[:3000]}\n\n"
        "Génère le plan d'analyse en JSON."
    )

    try:
        client = _get_client()
        call_kwargs: dict = dict(
            model=_model,
            messages=[
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_user},
            ],
        )
        # Utiliser response_format si disponible (modèles gpt-4o et suivants)
        _is_o_model = _model.startswith("o")
        if _is_o_model:
            call_kwargs["max_completion_tokens"] = 4096
        else:
            call_kwargs["max_tokens"] = 2048
            call_kwargs["temperature"] = 0.3
            try:
                call_kwargs["response_format"] = {"type": "json_object"}
            except Exception:
                pass

        response = client.chat.completions.create(**call_kwargs)
        raw = response.choices[0].message.content or ""

        # Parser le JSON — chercher le premier bloc { ... }
        raw_stripped = raw.strip()
        # Enlever les éventuels blocs markdown ```json ... ```
        if raw_stripped.startswith("```"):
            lines = raw_stripped.split("\n")
            raw_stripped = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            ).strip()

        data = json.loads(raw_stripped)

        steps = data.get("steps", [])
        if isinstance(steps, list) and steps:
            return steps
        return _PLAN_FALLBACK

    except Exception:
        return _PLAN_FALLBACK


# ─────────────────────────────────────────────────────────────────────────────
# Base de connaissances actuarielle (Knowledge Base)
# ─────────────────────────────────────────────────────────────────────────────
_KB_DIR = Path(__file__).parent / "Knowledge Base"


def load_knowledge_base_context(
    modules: list[str] | None = None,
    kb_dir: Path | None = None,
) -> str:
    """Charge la base de connaissances et retourne un bloc de doctrine formaté.

    Les fichiers JSON dans ``Knowledge Base/`` portent le même nom que les modules
    actuariels (p. ex. ``04_smoothing.json`` ↔ ``notebooks/04_smoothing.py``).
    Chaque fichier est un tableau de chunks : {id, source, section, type, tags, titre, contenu}.

    Args:
        modules: liste des noms de modules (sans extension) à charger.
                 ``None`` → charge tous les fichiers JSON disponibles.
        kb_dir:  Répertoire de la base de connaissances. ``None`` → utilise ``_KB_DIR``.

    Returns:
        Chaîne prête à être injectée dans ``{notebook_context}`` du system prompt.
        Chaîne vide si le répertoire n'existe pas ou si aucun fichier n'est trouvé.
    """
    effective_dir = kb_dir if kb_dir is not None else _KB_DIR
    if not effective_dir.exists():
        return ""

    if modules:
        json_files = [effective_dir / f"{m}.json" for m in modules
                      if (effective_dir / f"{m}.json").exists()]
    else:
        json_files = sorted(effective_dir.glob("*.json"))

    if not json_files:
        return ""

    sections: list[str] = []
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as fh:
                chunks = json.load(fh)
        except Exception:
            continue
        if not chunks:
            continue

        module_name = jf.stem
        lines = [f"\n[{module_name}]"]
        for chunk in chunks:
            titre = chunk.get("titre", "")
            contenu = chunk.get("contenu", "")
            if titre:
                lines.append(f"\n## {titre}")
            if contenu:
                lines.append(contenu)
        sections.append("\n".join(lines))

    if not sections:
        return ""

    header = (
        "DOCTRINE ACTUARIELLE — BASE DE CONNAISSANCES\n"
        "─────────────────────────────────────────────\n"
        "Les extraits suivants proviennent des notes méthodologiques de référence.\n"
        "Utilise-les pour guider tes choix méthodologiques et rédiger les justifications.\n"
    )
    return header + "".join(sections)


def search_knowledge_base(
    query: str,
    top_k: int = 5,
    kb_dir: Path | None = None,
) -> str:
    """Recherche par mots-clés dans la base de connaissances.

    Stratégie : score TF simplifié — pour chaque chunk, compte le nombre de termes
    de la requête présents dans (titre + contenu + tags + section).
    Retourne les ``top_k`` chunks les plus pertinents, formatés pour le LLM.

    Args:
        query:  Question ou thème à rechercher.
        top_k:  Nombre maximum de chunks à retourner.
        kb_dir: Répertoire de la base de connaissances. ``None`` → utilise ``_KB_DIR``.

    Returns:
        Chaîne formatée avec les chunks pertinents, ou message d'absence.
    """
    effective_dir = kb_dir if kb_dir is not None else _KB_DIR
    if not effective_dir.exists():
        return "[search_documentation] Base de connaissances introuvable."

    json_files = sorted(effective_dir.glob("*.json"))
    if not json_files:
        return "[search_documentation] Aucun fichier de doctrine disponible."

    # Normalisation de la requête : minuscules, mots de longueur ≥ 3
    terms = [t.lower() for t in query.replace("_", " ").split() if len(t) >= 3]

    scored: list[tuple[int, str, dict]] = []  # (score, module_name, chunk)
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as fh:
                chunks = json.load(fh)
        except Exception:
            continue
        module_name = jf.stem
        for chunk in chunks:
            searchable = " ".join([
                chunk.get("titre", ""),
                chunk.get("contenu", ""),
                chunk.get("section", ""),
                " ".join(chunk.get("tags", [])),
            ]).lower()
            score = sum(1 for t in terms if t in searchable)
            if score > 0:
                scored.append((score, module_name, chunk))

    if not scored:
        return (
            f"[search_documentation] Aucun résultat pour « {query} ».\n"
            "Procède avec ton jugement d'expert actuariel."
        )

    # Tri décroissant par score, puis top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    lines = [f"[search_documentation] Résultats pour « {query} » ({len(top)}/{len(scored)} chunks) :"]
    for score, module, chunk in top:
        titre = chunk.get("titre", "(sans titre)")
        contenu = chunk.get("contenu", "")
        lines.append(f"\n--- [{module}] {titre} (pertinence : {score}) ---")
        lines.append(contenu)

    return "\n".join(lines)


try:
    from actuary_logger import LOGGER as _TOOL_LOGGER
except ImportError:
    class _NoLogger:
        def log(self, *a, **k): pass
    _TOOL_LOGGER = _NoLogger()

load_dotenv()

try:
    from actuarial_params import PARAMS as _PARAMS
    MAX_ITERATIONS = _PARAMS["agent"]["max_iterations"]
except Exception:
    MAX_ITERATIONS = 40  # fallback si actuarial_params non disponible
MAX_OUTPUT_LENGTH = 3000  # Troncature des sorties pour ne pas saturer la fenêtre
                           # contextuelle du LLM avec des logs trop volumineux.

# ─────────────────────────────────────────────────────────────────────────────
# Outil exposé à l'agent
# ─────────────────────────────────────────────────────────────────────────────
# TOOLS est la liste des outils au format OpenAI function-calling.
# Chaque entrée décrit au LLM CE QU'IL PEUT FAIRE et DANS QUEL CAS l'utiliser.
# Le champ "description" est critique : c'est lui qui guide la décision du LLM
# sur quel outil appeler (et quand NE PAS l'appeler).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Exécute une cellule de code Python dans le noyau partagé. "
                "Le résultat (stdout et erreurs) te sera retourné pour que tu puisses adapter la suite."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Code Python à exécuter. Peut utiliser les variables définies dans les appels précédents.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Phrase en français décrivant ce que fait cette étape (affichée à l'utilisateur à la place du code).",
                    },
                },
                "required": ["code", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documentation",
            "description": (
                "Recherche dans la base documentaire actuarielle (notes méthodologiques, "
                "rapports passés, décisions de comité technique). "
                "À utiliser AVANT toute décision de jugement : choix d'hypothèse "
                "méthodologique, interprétation d'une anomalie (SMR anormal, "
                "non-monotonicité persistante, données atypiques), "
                "rédaction d'une justification dans le rapport final. "
                "Ne PAS appeler pour des calculs numériques."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question ou thème à rechercher dans la documentation.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "Pose une question à l'utilisateur humain et attend sa réponse avant de continuer. "
                "À utiliser UNIQUEMENT quand une décision méthodologique nécessite une validation humaine : "
                "lisseur proche (auto_select_smoother status='close'), SMR hors [0.3, 3.0], "
                "choix entre méthodes statistiquement équivalentes. "
                "NE PAS appeler pour des informations, des calculs ou des questions rhétoriques."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question claire et concise à poser à l'utilisateur.",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Réponses suggérées (ex: ['Whittaker', 'Gompertz']). Optionnel.",
                    },
                },
                "required": ["question"],
            },
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """\
Tu es un agent actuariel expert en construction de tables de mortalité d'expérience.

OBJECTIF
────────
Construire la meilleure table de mortalité pour le portefeuille fourni.
Tu décides toi-même de la méthode, de l'ordre des étapes et du nombre d'itérations.
Tu t'arrêtes quand tu as une table validée statistiquement et documentée.

BIBLIOTHÈQUE DE FONCTIONS DISPONIBLES
──────────────────────────────────────
Toutes ces fonctions sont disponibles dans le kernel sous forme de modules.
Appelle-les directement — tu n'as pas besoin de les réécrire.

MODULE data_prep  (01_data_preparation.py)
  data_prep.load_data(path, encoding='utf-8', column_mapping=COLUMN_MAPPING, value_mapping=VALUE_MAPPING)
    QUAND : toujours en premier. Charge CSV ou Excel.
    Si COLUMN_MAPPING ou VALUE_MAPPING sont non-vides dans le kernel, les passer OBLIGATOIREMENT
    pour normaliser les noms de colonnes et les valeurs catégorielles non-standard.
    RETOURNE : (DataFrame, summary_dict)

  data_prep.generate_synthetic_data(n=50000, sexe='H', seed=42)
    QUAND : aucun fichier fourni — génère des données de test.
    RETOURNE : DataFrame

  data_prep.clean_data(df, date_fin_observation=DATE_FIN_OBSERVATION)
    QUAND : après chargement. Valide cohérence des dates, âges, cause_sortie.
    IMPORTANT : passer TOUJOURS date_fin_observation=DATE_FIN_OBSERVATION pour que
    les individus encore actifs (date_sortie sentinelle 31/12/2999 → NaT) soient
    conservés et traités comme censurés à la fin d'observation.
    RETOURNE : (df_clean, rapport_dict)

  data_prep.compute_ages(df)
    QUAND : après clean_data. Calcule age_entree, age_sortie, duree_obs_ans.
    RETOURNE : DataFrame enrichi

  data_prep.detect_anomalies(df)
    QUAND : après compute_ages. Détecte anomalies structurelles.
    RETOURNE : dict {duplicates, missing_values, severity, recommendations}

MODULE exposure  (02_exposure.py)
  exposure.compute_exposure_by_age(df, age_min=20, age_max=90)
    QUAND : toujours, après préparation des données.
    RETOURNE : DataFrame [age, E_x, D_x, mu_x, q_x_brut]

  exposure.exposure_summary(exposure_table)
    QUAND : après compute_exposure_by_age pour un résumé rapide.
    RETOURNE : dict {total_exposure, total_deaths, pct_low_credibility, ...}

MODULE crude_rates  (03_crude_rates.py)
  crude_rates.crude_rates_central(exposure_table)
    QUAND : par défaut — exposition en années-personnes.
    RETOURNE : DataFrame avec colonne qx

  crude_rates.crude_rates_binomial(exposure_table)
    QUAND : exposition initiale disponible (nombre de têtes début d'année).
    RETOURNE : DataFrame avec colonne qx

  crude_rates.crude_rates_kaplan_meier(df, age_min, age_max)
    QUAND : petits effectifs (< 5 000 contrats) — estimateur non-paramétrique.
    RETOURNE : DataFrame avec colonne qx

MODULE smoothing  (04_smoothing.py)
  Toutes les fonctions de lissage prennent exposure_table (avec q_x_brut)
  et retournent dict {ages, qx_smoothed, method, params, n_non_monotone_after_40}

  smoothing.smooth_whittaker(qx_table, lambda_wh=PARAMS["smoothing"]["lambda_wh"])
    QUAND : données denses, pas d'hypothèse paramétrique.

  smoothing.smooth_gompertz(qx_table, age_min_fit=PARAMS["smoothing"]["gompertz_age_min"])
    QUAND : > 20 % des âges avec exposition faible. Extrapolation aux grands âges.

  smoothing.smooth_makeham(qx_table, age_min_fit=PARAMS["smoothing"]["makeham_age_min"])
    QUAND : mortalité accidentelle significative (jeunes âges présents).

  smoothing.smooth_spline(qx_table)
    QUAND : données très denses, flexibilité locale souhaitée.

MODULE smoothing_selector  (smoothing_selector.py)
  smoothing_selector.auto_select_smoother(qx_table, exposure_table)
    QUAND : TOUJOURS à l'étape 5 — sélection automatique du meilleur modèle.
    RETOURNE : dict {status, best_method, best_result, comparison_df, reason, ...}
    STATUS POSSIBLES :
      "clear"   → gagnant net — utilise best_result et continue
      "close"   → deux modèles proches — ARRÊTE et présente comparison_df à l'utilisateur
      "escalate"→ problème monotonicité ou convergence — ARRÊTE et signale le problème

  smoothing_selector.print_selection_result(result)
    Affiche un résumé lisible du résultat (utile pour le diagnostic).

MODULE diagnostics  (05_diagnostics.py)
  diagnostics.diagnose_credibility(exposure_table, threshold=10)
    QUAND : OBLIGATOIRE avant tout lissage.
    RETOURNE : dict {pct_low, recommendation: 'parametric'|'non-parametric', recommendation_reason}

  diagnostics.diagnose_monotonicity(qx_series, age_series)
    QUAND : après lissage, pour vérifier la monotonie.
    RETOURNE : dict {n_violations, violation_ages, is_monotone}

  diagnostics.compare_smoothers(smoothers_dict, exposure_table)
    QUAND : après ≥ 2 lissages pour choisir le meilleur.
    smoothers_dict = {'Whittaker': wh_result, 'Gompertz': gom_result, ...}
    RETOURNE : (DataFrame comparaison AIC/MSE/monotonie, dict {recommended, reason})

  diagnostics.compute_smr(exposure_table, qx_col=None, sexe='H')
    QUAND : après lissage, pour comparer à la référence TH/TF 00-02.
    RETOURNE : dict {smr_global, ci_lower, ci_upper, d_observed, d_expected, interpretation}

MODULE validation  (06_validation.py)
  validation.confidence_intervals(exposure_table, alpha=0.05)
    QUAND : après lissage, pour quantifier l'incertitude.
    RETOURNE : DataFrame [age, qx, ci_lower, ci_upper]

  validation.chi_square_test(exposure_table)
    QUAND : test formel d'adéquation observé/attendu.
    RETOURNE : dict {statistic, p_value, conclusion}

  validation.prudence_margin(exposure_table)
    QUAND : pour les produits d'assurance-vie/rentes — vérifie la prudence.
    RETOURNE : dict {prudence_level: 'insufficient'|'adequate'|'conservative', ...}

  validation.cox_model(df, covariates=['sexe'])
    QUAND : variable sexe ou produit disponible — analyse des différentiels.
    RETOURNE : dict {hazard_ratios, p_values, interpretation}

MODULE benchmarking  (07_benchmarking.py)
  benchmarking.load_reference_table(name='TH0002', sexe='H')
    Tables disponibles : 'TH0002', 'TF0002', 'TD8890', 'TPRV93'
    RETOURNE : DataFrame [age, qx_ref]

  benchmarking.abatement_factors(exposure_table)
    QUAND : pour quantifier l'écart au référentiel age par age.
    RETOURNE : (DataFrame [age, qx_exp, qx_ref, abatement_factor], summary_dict)

  benchmarking.logit_regression(exposure_table)
    QUAND : facteurs d'abattement non constants selon l'âge.
    RETOURNE : dict {a, b, r_squared, interpretation}

  benchmarking.export_table(exposure_table, file_path=None, sexe='H', smr=None)
    QUAND : toujours en dernier — exporte la table finale.
    RETOURNE : chemin du fichier CSV créé

MODULE visualization  (08_visualization.py)
  Toutes les fonctions retournent des bytes PNG. matplotlib les affiche automatiquement.

  visualization.plot_exposure_by_age(exposure_table)
  visualization.plot_crude_vs_smoothed(exposure_table, smoothed_dict)
    smoothed_dict = {'Whittaker': qx_array, 'Gompertz': qx_array}
  visualization.plot_smr_by_age(smr_result)
  visualization.plot_confidence_bands(exposure_table, ci_result=None)
  visualization.plot_survival_curve(exposure_table)
  visualization.plot_observed_vs_expected(exposure_table)

COMPLÉTUDE MINIMALE — ce qu'un actuaire ferait toujours
────────────────────────────────────────────────────────
Les étapes suivantes sont NON NÉGOCIABLES. Tu décides librement de l'ordre,
du découpage des appels et de la façon de gérer les anomalies — mais tu ne
peux pas conclure sans avoir couvert chacun de ces points :

  □ Chargement et nettoyage des données (load_data, clean_data, compute_ages)
  □ Calcul des expositions (compute_exposure_by_age)
  □ Calcul des taux bruts (crude_rates_central ou autre selon le portefeuille)
  □ Diagnostic de crédibilité (diagnose_credibility)
  □ Sélection et application du lissage (auto_select_smoother)
      → "close"   : ARRÊTE et demande confirmation humaine avant de continuer
      → "escalate": ARRÊTE et explique le problème à l'utilisateur
  □ Validation statistique : intervalles de confiance + test du chi-deux (chi_square_test)
  □ SMR (compute_smr) et comparaison à une table de référence (benchmarking.load_reference_table)
  □ Backtesting O/A : tableau décès observés vs modélisés par décennie d'âge
      → construire t7 (Tableau 7 — Étape 7) avec D_obs, D_exp, écart, rapport O/A
      → visualization.plot_observed_vs_expected(exposure_table)  ← OBLIGATOIRE
  □ Graphiques obligatoires : plot_crude_vs_smoothed + plot_smr_by_age + plot_observed_vs_expected
  □ Export de la table finale (benchmarking.export_table)
  □ Synthèse avec justification de chaque choix méthodologique

ÉTAPES CONDITIONNELLES — à faire si pertinent
──────────────────────────────────────────────
  • detect_anomalies()        — si les données semblent hétérogènes
  • prudence_margin()         — si le rapport est destiné à un bilan prudentiel
  • cox_model()               — si la variable sexe ou produit est disponible
  • logit_regression()        — si les facteurs d'abattement varient avec l'âge
  • compute_exposure_by_year()— si une dérive temporelle est suspectée
  • Graphiques supplémentaires— selon les anomalies observées

TABLEAUX ET VISUELS OBLIGATOIRES
─────────────────────────────────
À 8 moments clés, tu DOIS produire un tableau de synthèse ET (si pertinent) un graphique.
Tu choisis librement la mise en forme, mais ces 8 rendus ne sont pas optionnels.
Utilise ces patterns de code :

  Règle d'affichage CRITIQUE : utilise TOUJOURS display(df) ou laisse le DataFrame
  en dernière expression de la cellule — JAMAIS print() ni .to_string() qui produisent
  du texte brut sans rendu tableau dans Jupyter.

  1. APRÈS load_data — aperçu du fichier chargé :
     display(pd.DataFrame({
         'Indicateur': ['Lignes', 'Colonnes disponibles', 'Première date entrée', 'Dernière date entrée'],
         'Valeur': [len(df), len(df.columns),
                    str(df['date_entree'].min().date()), str(df['date_entree'].max().date())]
     }))

  2. APRÈS clean_data — tableau des suppressions :
     display(pd.DataFrame(rapport['removal_reasons'].items(),
                          columns=['Raison', 'N supprimés']).set_index('Raison'))

  3. APRÈS compute_ages — distribution des âges (pyramide synthétique) :
     display(df.groupby(pd.cut(df['age'], bins=range(20,96,5)))
               .agg(N=('age','count'), pct=('age', lambda x: 100*len(x)/len(df)))
               .round(1).rename_axis('Tranche d\'âge'))

  4. APRÈS compute_exposure_by_age — top âges + graphique d'exposition :
     display(exposure_table.sort_values('E_x', ascending=False)
             .head(20)[['age','E_x','D_x','q_x_brut']].round(4).reset_index(drop=True))
     visualization.plot_exposure_by_age(exposure_table)  ← OBLIGATOIRE (style page 6 du rapport)

  5. APRÈS lissage — comparatif méthodes (construis ce dict au fil des tests) :
     display(pd.DataFrame(results_list,   # [{'méthode':'Whittaker','lambda':100,'AIC':…,'violations':4},…]
                          columns=['méthode','paramètre','AIC','violations_mono','RMSE']).round(3))
     visualization.plot_crude_vs_smoothed(exposure_table, smoothed_dict)  ← OBLIGATOIRE

  6. APRÈS compute_smr — SMR décennal :
     display(pd.DataFrame(smr_result.get('by_decade', {}))
             [['decade','D_obs','D_exp','SMR','IC_inf','IC_sup']].round(3))

  7. TABLE DE SYNTHÈSE FINALE (Tableau 7 — format rapport professionnel) :
     # Construire AVANT export_table — colonnes exactes du standard professionnel TD :
     expo_tot = exposure_table['E_x'].sum()
     t7 = exposure_table[['age','E_x','D_x','D_exp','qx_lisse','IC_inf','IC_sup']].copy()
     t7['proportion'] = (t7['E_x'] / expo_tot * 100).round(2)
     t7['ecart'] = t7['D_x'] - t7['D_exp']
     t7['rapport_OA'] = (t7['D_x'] / t7['D_exp'].replace(0, float('nan'))).round(3)
     display(t7[['age','E_x','proportion','D_x','D_exp','ecart','rapport_OA','IC_inf','IC_sup']]
               .rename(columns={'E_x':'Exposition','proportion':'Proportion (%)','D_x':'D_obs',
                                 'D_exp':'D_exp','ecart':'Écart','rapport_OA':'Rapport O/A',
                                 'IC_inf':'IC Min 95%','IC_sup':'IC Max 95%'})
               .round(3).reset_index(drop=True))

  8. FIGURE DÉCÈS OBSERVÉS VS MODÉLISÉS (Figure 8 — format rapport professionnel) :
     visualization.plot_observed_vs_expected(exposure_table)  ← OBLIGATOIRE en fin d'analyse

AUTO-VÉRIFICATION AVANT CONCLUSION
────────────────────────────────────
Avant de rédiger ta synthèse finale, parcours mentalement la checklist ci-dessus.
Pour chaque case non cochée : soit tu la complètes maintenant, soit tu expliques
explicitement pourquoi elle ne s'applique pas à ce portefeuille.

RÈGLES DE CODE
──────────────
- Chaque appel execute_python doit faire UNE SEULE chose (charger, nettoyer, calculer, OU afficher).
- Maximum 40 lignes de code Python par appel. Si tu as besoin de plus, découpe en plusieurs appels.
- Ne mets JAMAIS de commentaires longs dans des chaînes de caractères (titres plt, labels…).
  Utilise des variables intermédiaires : titre = "Mon titre court" puis plt.title(titre).
- N'écris jamais de texte libre à l'intérieur d'une chaîne Python.
- BIBLIOTHÈQUES AUTORISÉES UNIQUEMENT : pandas, numpy, scipy, matplotlib, seaborn, pathlib, json, datetime.
  N'importe JAMAIS statsmodels, lifelines, scikit-learn ou toute autre bibliothèque externe.
- Toutes les fonctions actuarielles sont décrites dans la section BIBLIOTHÈQUE ci-dessus.
  N'implémente JAMAIS toi-même un calcul déjà couvert par ces fonctions.
  Utilise EXCLUSIVEMENT les noms de modules tels qu'ils apparaissent dans le kernel :
  data_prep, exposure, crude_rates, smoothing, diagnostics, validation, benchmarking, visualization, smoothing_selector.
- CHEMINS : le répertoire de travail est la RACINE DU PROJET (pas le dossier notebooks/).
  Utilise uniquement FILE_PATH pour les données d'entrée — ne construis jamais de chemin absolu.
  N'ajoute JAMAIS rien à sys.path et n'appelle JAMAIS os.chdir() : les modules sont déjà chargés.
  Pour les sorties (export_table, etc.), utilise des chemins relatifs comme "outputs/table.csv".

RÈGLES MÉTIER
─────────────
- Justifie chaque choix méthodologique dans le champ `description` de chaque appel.
- Appelle search_documentation AVANT toute décision de jugement :
    • choix d'un modèle (si auto_select_smoother retourne "close")
    • interprétation d'un SMR anormal (< 0.80 ou > 1.20)
    • rédaction d'une justification dans la synthèse finale
  Ne l'appelle PAS pour des calculs numériques.
- Si n_contrats < 5 000 → préfère crude_rates_kaplan_meier.
- Si SMR hors [0.3, 3.0] → anomalie probable — vérifier les données.
- Ne jamais conclure sans validation statistique (chi-deux ou IC).
- PARAMÈTRES MÉTIER : tous les seuils sont dans PARAMS (variable disponible dans le kernel).
    Exemples : PARAMS["smr"]["lower"], PARAMS["credibility"]["threshold_low"]
    Ne hardcode jamais un seuil numérique — lis-le toujours depuis PARAMS.

INTERACTIONS UTILISATEUR
────────────────────────
ask_user(question, options=[])
  QUAND utiliser : décision méthodologique qui nécessite une validation humaine.
    • auto_select_smoother retourne status='close' → demande lequel préférer
    • SMR hors [0.3, 3.0] → demande si l'utilisateur veut continuer malgré l'anomalie
    • Choix entre méthodes statistiquement équivalentes → demande la préférence
  NE PAS utiliser pour : informations factuelles, calculs, questions rhétoriques.
  L'agent est mis en pause jusqu'à réception de la réponse dans le chat RAG.

{notebook_context}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Client OpenAI
# ─────────────────────────────────────────────────────────────────────────────
def _get_client() -> OpenAI:
    """Crée et retourne un client OpenAI authentifié depuis la variable d'environnement.

    On instancie le client à chaque appel plutôt que de le mettre en cache global
    pour éviter des problèmes de timeout lors de sessions longues (> 30 min).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Clé API OpenAI manquante. Vérifiez votre fichier .env (OPENAI_API_KEY=sk-...).")
    return OpenAI(api_key=api_key)


def _truncate(text: str) -> str:
    """Tronque la sortie d'outil à MAX_OUTPUT_LENGTH caractères.

    Stratégie différenciée :
    - Sortie normale  → garde le DÉBUT (en-têtes, premières lignes du DataFrame)
    - Erreur/traceback → garde la FIN  (où se trouve le message d'erreur utile)
      + les 200 premiers caractères pour conserver le contexte de l'appel.
    """
    if len(text) <= MAX_OUTPUT_LENGTH:
        return text
    is_error = text.startswith("❌")
    if is_error:
        # Pour un traceback, l'information utile est toujours à la fin.
        # On conserve aussi le début pour que le LLM sache quelle cellule a échoué.
        head = text[:200]
        tail = text[-(MAX_OUTPUT_LENGTH - 200):]
        return f"{head}\n... [tronqué — {len(text)} caractères au total] ...\n{tail}"
    return text[:MAX_OUTPUT_LENGTH] + f"\n... [tronqué — {len(text)} caractères au total]"


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic automatique des erreurs courantes
# ─────────────────────────────────────────────────────────────────────────────

def _diagnose_error(output_text: str) -> str:
    """Analyse le traceback et retourne un hint contextuel pour l'agent.

    Évite que l'agent reçoive un message générique "corrige ton code" sans
    indice sur la cause réelle — ce qui lui ferait répéter la même erreur.
    """
    t = output_text

    if "SyntaxError" in t:
        return (
            "SyntaxError : le code généré est incomplet ou mal formé (chaîne non fermée, "
            "parenthèse manquante…). Cause probable : le bloc de code était trop long et "
            "a été tronqué. Découpe ce bloc en 2-3 étapes distinctes plus courtes "
            "(max ~50 lignes par appel execute_python)."
        )

    if "KeyError" in t:
        import re
        key = re.search(r"KeyError:\s*(.+)", t)
        key_val = key.group(1).strip() if key else "?"
        if "Timestamp" in key_val:
            return (
                f"KeyError sur {key_val} : tu utilises un Timestamp comme nom de colonne. "
                "Pour filtrer sur une date, écris df[df['date_sortie'] <= DATE_FIN_OBSERVATION], "
                "pas df[DATE_FIN_OBSERVATION]."
            )
        return (
            f"KeyError sur {key_val} : la colonne ou la clé n'existe pas. "
            "Vérifie les colonnes disponibles avec df.columns ou dict.keys()."
        )

    if "UnicodeDecodeError" in t or "codec can't decode" in t.lower():
        return (
            "Erreur d'encodage CSV. Relance avec encoding='latin-1' ou encoding='cp1252' "
            "à la place de 'utf-8'."
        )

    if "FileNotFoundError" in t or "No such file" in t:
        return (
            "Fichier introuvable. Utilise la variable FILE_PATH déjà définie dans le kernel "
            "plutôt qu'un chemin en dur."
        )

    if "AttributeError" in t:
        import re
        attr = re.search(r"AttributeError: (.+)", t)
        return (
            f"AttributeError : {attr.group(1).strip() if attr else '?'}. "
            "Vérifie que le module est bien chargé et que tu appelles la bonne méthode."
        )

    if "ValueError" in t and "could not convert" in t.lower():
        return (
            "Erreur de conversion de type. Vérifie que les colonnes de dates sont bien "
            "au format datetime (pd.to_datetime) avant tout calcul."
        )

    if "convergence" in t.lower() or "ConvergenceWarning" in t:
        return (
            "Le modèle n'a pas convergé. Essaie de réduire les bornes d'âge "
            "(age_min_fit, age_max_fit) ou d'utiliser un modèle plus simple (Whittaker)."
        )

    # Hint générique si aucun pattern reconnu
    return "Analyse le traceback complet ci-dessus et relance avec du code corrigé."


# ─────────────────────────────────────────────────────────────────────────────
# Boucle ReAct principale
# ─────────────────────────────────────────────────────────────────────────────
def run_agent_loop(
    user_message: str,
    notebook_context: str,
    conversation_history: list,
    execute_fn: Callable[[str], tuple],
    system_prompt_template: str = None,
    max_steps: int = None,
    wait_for_user_fn: Callable[[str, list], str] | None = None,
    kb_dir: Path | None = None,
) -> Generator[dict, None, None]:
    """Boucle ReAct : appelle OpenAI, exécute les outils, renvoie les résultats.

    Args:
        user_message:          Message courant de l'utilisateur.
        notebook_context:      Contenu des notebooks injecté dans le system prompt.
        conversation_history:  Historique AVANT le message courant.
        execute_fn:            Callable(code: str) -> (output_text: str, figures: list[bytes])
        max_steps:             Nombre maximum d'appels d'outils avant de s'arrêter.
                               None = utilise MAX_ITERATIONS (mode normal).
                               1 = mode pas-à-pas : s'arrête après chaque étape.
        wait_for_user_fn:      Callable(question: str, options: list) -> str  bloquant.
                               Si fourni, le tool ask_user appelle cette fonction et attend la
                               réponse avant de reprendre. Si None, retourne "continuer" immédiatement.

    Yields:
        {"type": "step",     "description": str, "code": str, "output": str, "figures": list[bytes]}
        {"type": "question", "content": str, "options": list}
        {"type": "summary",  "content": str}
        {"type": "history",  "messages": list}
        {"type": "error",    "content": str}
    """
    client = _get_client()
    template = system_prompt_template if system_prompt_template is not None else SYSTEM_PROMPT_TEMPLATE
    system_prompt = template.replace("{notebook_context}", notebook_context)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    # Les modèles de la série "o" (o1, o3…) utilisent max_completion_tokens
    # au lieu de max_tokens, et ne supportent pas le paramètre temperature.
    _is_o_model = config.REASONING_MODEL.startswith("o")
    _steps_done = 0
    _limit = max_steps if max_steps is not None else MAX_ITERATIONS

    for _iter in range(MAX_ITERATIONS):
        # ── Événement "thinking" : indique à l'UI ce que l'agent s'apprête à faire
        if _iter == 0:
            _thinking_msg = "Analyse de la demande et des données disponibles…"
        else:
            _last_role = messages[-1].get("role", "") if messages else ""
            if _last_role == "tool":
                _thinking_msg = "Analyse du résultat et décision de la prochaine action…"
            else:
                _thinking_msg = "Réflexion en cours…"
        yield {"type": "thinking", "message": _thinking_msg}

        call_kwargs = dict(
            model=config.REASONING_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        if _is_o_model:
            call_kwargs["max_completion_tokens"] = config.MAX_COMPLETION_TOKENS
        else:
            call_kwargs["max_tokens"] = config.MAX_TOKENS
            call_kwargs["temperature"] = config.TEMPERATURE

        response = client.chat.completions.create(**call_kwargs)

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        message = choice.message

        # ── L'agent veut exécuter du code ────────────────────────────────────
        # On ajoute le message assistant AVANT les résultats d'outils pour
        # respecter la structure attendue par l'API OpenAI :
        #   assistant (avec tool_calls) → tool → tool → ... → assistant → ...
        if finish_reason == "tool_calls" and message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")

                if tool_name == "execute_python":
                    code = args.get("code", "")
                    description = args.get("description", "")
                    output_text, figures = execute_fn(code)

                    # Journal structuré pour le RAG : nom de l'outil + description
                    _TOOL_LOGGER.log(
                        "agent:execute_python",
                        description or "(code sans description)",
                        {"n_chars_code": len(code), "success": "❌" not in output_text},
                    )

                    yield {
                        "type": "step",
                        "description": description,
                        "code": code,
                        "output": output_text,
                        "figures": figures,
                    }

                    tool_content = _truncate(output_text)
                    if "❌ Erreur" in output_text:
                        tool_content += "\n\nATTENTION : erreur ci-dessus. " + _diagnose_error(output_text)

                    # Mode pas-à-pas : s'arrêter après max_steps appels d'outils
                    _steps_done += 1
                    if _steps_done >= _limit:
                        yield {"type": "history", "messages": messages[1:]}  # sans system
                        yield {
                            "type": "summary",
                            "content": (
                                f"**Étape exécutée.** *(mode pas-à-pas — {_steps_done}/{_limit})*\n\n"
                                "Réponds **continuer** pour l'étape suivante, "
                                "ou pose une question sur ce que l'agent vient de faire."
                            ),
                        }
                        return

                elif tool_name == "search_documentation":
                    query = args.get("query", "")
                    _TOOL_LOGGER.log(
                        "agent:search_documentation",
                        f"Recherche documentaire : {query}",
                        {"query": query},
                    )
                    output_text = search_knowledge_base(query, kb_dir=kb_dir)
                    figures = []

                    yield {
                        "type": "step",
                        "description": f"Recherche documentation : {query}",
                        "output": output_text,
                        "figures": figures,
                    }

                    tool_content = output_text

                elif tool_name == "ask_user":
                    question_text = args.get("question", "")
                    options = args.get("options", [])

                    # Signal pour l'UI (le polling injectera la question dans le RAG chat)
                    yield {
                        "type": "question",
                        "content": question_text,
                        "options": options,
                    }

                    # Bloquer le thread jusqu'à réception de la réponse utilisateur
                    # (wait_for_user_fn est fourni par canvas_app via _make_wait_for_user_fn)
                    if wait_for_user_fn is not None:
                        user_reply = wait_for_user_fn(question_text, options)
                    else:
                        user_reply = "continuer"

                    _TOOL_LOGGER.log(
                        "agent:ask_user",
                        f"Question posée : {question_text[:80]}",
                        {"question": question_text, "reply": user_reply},
                    )

                    output_text = f"[Réponse utilisateur] {user_reply}"
                    tool_content = output_text
                    figures = []

                else:
                    output_text = f"Outil inconnu : {tool_name}"
                    figures = []
                    yield {
                        "type": "step",
                        "description": f"Outil non reconnu : {tool_name}",
                        "output": output_text,
                        "figures": figures,
                    }
                    tool_content = output_text

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_content,
                })

        # ── L'agent a terminé (réponse texte finale) ─────────────────────────
        # "stop"   → le LLM a fini de répondre normalement.
        # "length" → réponse tronquée par max_tokens : on l'accepte quand même
        #            pour ne pas bloquer l'utilisateur (la synthèse sera partielle).
        elif finish_reason in ("stop", "length"):
            final_content = message.content or ""
            yield {"type": "summary", "content": final_content}
            messages.append({"role": "assistant", "content": final_content})
            # On exclut le system prompt de l'historique exporté : il est reconstruit
            # à chaque appel depuis le template, donc inutile de le stocker côté UI.
            yield {"type": "history", "messages": messages[1:]}  # sans system prompt
            return

        # ── Cas inattendu ─────────────────────────────────────────────────────
        else:
            yield {
                "type": "error",
                "content": f"Fin inattendue de l'API : finish_reason={finish_reason}",
            }
            return

    yield {
        "type": "error",
        "content": f"L'agent a dépassé la limite de {MAX_ITERATIONS} itérations sans terminer.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers pour la boucle encodeur-décodeur
# ─────────────────────────────────────────────────────────────────────────────

def make_isolated_kernel() -> dict:
    """Crée un kernel Python isolé, utilisable dans la boucle encodeur.

    Équivalent à workflow_executor.make_kernel() mais clairement nommé
    pour signaler qu'il ne partage pas l'état avec l'agent actuariel principal.
    Chaque appel retourne un namespace indépendant.
    """
    from workflow_executor import make_kernel as _mk
    return _mk()


def run_agent_on_synthetic(
    system_prompt: str,
    kernel: dict | None = None,
    n: int = 50_000,
    sexe: str = "H",
    seed: int = 42,
) -> tuple[list[dict], str]:
    """Lance run_agent_loop sur des données synthétiques.

    Utilisé par la boucle d'optimisation de l'encodeur pour évaluer la qualité
    du prompt sans données réelles. Le prompt est universel (structure + méthode) —
    les valeurs numériques varient mais la structure du rapport doit rester identique.

    Args:
        system_prompt: Section MISSION + section technique à évaluer.
        kernel:        Namespace Python isolé. None → make_isolated_kernel().
        n:             Nombre de contrats synthétiques (50 000 par défaut).
        sexe:          'H' ou 'F'.
        seed:          Graine pour la reproductibilité.

    Returns:
        (steps, summary) où steps est la liste des events "step"
        et summary est la synthèse finale de l'agent.
    """
    import tempfile, os as _os, pandas as _pd

    if kernel is None:
        kernel = make_isolated_kernel()

    # Générer des données synthétiques et les écrire dans un fichier CSV temporaire
    data_prep = kernel.get("data_prep")
    if data_prep is None or not hasattr(data_prep, "generate_synthetic_data"):
        raise RuntimeError(
            "Module data_prep non chargé dans le kernel — "
            "vérifiez que make_isolated_kernel() charge correctement les modules."
        )

    synth_df = data_prep.generate_synthetic_data(n=n, sexe=sexe, seed=seed)

    tmp_dir = Path(__file__).parent / "uploads"
    tmp_dir.mkdir(exist_ok=True)
    tmp_csv = tmp_dir / f"_synthetic_{seed}_{n}_{sexe}.csv"
    synth_df.to_csv(tmp_csv, index=False, encoding="utf-8")

    kernel["FILE_PATH"] = str(tmp_csv)
    kernel["SEXE"] = sexe

    # Message utilisateur minimal pour démarrer l'analyse
    user_message = (
        f"Analyse le portefeuille synthétique ({n:,} contrats, sexe {sexe}) "
        f"disponible dans FILE_PATH. Construis la table de mortalité complète "
        f"selon les instructions du prompt et produis tous les livrables demandés."
    )

    from notebook_runner import execute_cell as _exec_cell
    from workflow_executor import capture_figures as _cap_figs

    def _execute_fn(code: str) -> tuple:
        output = _exec_cell(code, kernel)
        figs = _cap_figs(kernel)
        return output, figs

    steps: list[dict] = []
    summary: str = ""

    for event in run_agent_loop(
        user_message=user_message,
        notebook_context="",
        conversation_history=[],
        execute_fn=_execute_fn,
        system_prompt_template=system_prompt,
    ):
        if event.get("type") == "step":
            steps.append(event)
        elif event.get("type") == "summary":
            summary = event.get("content", "")

    return steps, summary
