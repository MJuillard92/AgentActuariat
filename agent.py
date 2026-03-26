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

import json
import os
from typing import Callable, Generator

from dotenv import load_dotenv
from openai import OpenAI

import config

try:
    from actuary_logger import LOGGER as _TOOL_LOGGER
except ImportError:
    class _NoLogger:
        def log(self, *a, **k): pass
    _TOOL_LOGGER = _NoLogger()

load_dotenv()

MAX_ITERATIONS = 25    # Limite de sécurité : évite une boucle infinie si le LLM
                       # ne converge pas (p. ex. erreur répétée non récupérable).
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
  data_prep.load_data(path, encoding='utf-8')
    QUAND : toujours en premier. Charge CSV ou Excel.
    RETOURNE : (DataFrame, summary_dict)

  data_prep.generate_synthetic_data(n=50000, sexe='H', seed=42)
    QUAND : aucun fichier fourni — génère des données de test.
    RETOURNE : DataFrame

  data_prep.clean_data(df)
    QUAND : après chargement. Valide cohérence des dates, âges, cause_sortie.
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
  □ Validation statistique : intervalles de confiance + test du chi-deux
  □ SMR (compute_smr) et comparaison à une table de référence
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

    Yields:
        {"type": "step",    "description": str, "code": str, "output": str, "figures": list[bytes]}
        {"type": "summary", "content": str}
        {"type": "history", "messages": list}
        {"type": "error",   "content": str}
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

    for _ in range(MAX_ITERATIONS):
        call_kwargs = dict(
            model=config.REASONING_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        if _is_o_model:
            call_kwargs["max_completion_tokens"] = config.MAX_TOKENS
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
                    # Connecteur RAG documentaire non encore activé.
                    # On retourne un message de substitution pour que le LLM puisse
                    # continuer avec son jugement d'expert plutôt que de bloquer.
                    query = args.get("query", "")
                    _TOOL_LOGGER.log(
                        "agent:search_documentation",
                        f"Recherche documentaire : {query}",
                        {"query": query},
                    )
                    output_text = (
                        f"[search_documentation] Requête : « {query} »\n"
                        "Base documentaire non disponible — ce connecteur sera activé ultérieurement.\n"
                        "Procède avec ton jugement d'expert actuariel et documente "
                        "ton raisonnement dans le champ description de la prochaine étape."
                    )
                    figures = []

                    yield {
                        "type": "step",
                        "description": f"Recherche documentation : {query}",
                        "output": output_text,
                        "figures": figures,
                    }

                    tool_content = output_text

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
