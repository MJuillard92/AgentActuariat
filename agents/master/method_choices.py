"""
agents/master/method_choices.py
Découverte générique des paramètres `method` (multi-valeurs) des tools
appelés par le Builder selon le mode courant, ET orchestration complète
de la désambiguation conversationnelle avec l'utilisateur.

Source de vérité = docstring `INPUTS.params.<name>.values: A | B | C`
parsée par le catalogue des tools (cf. tools/catalogue.py).

Interface publique (découverte) :
    method_choices_for_mode(report_mode, gender_segmentation, study_plan)
        → list[ChoiceSpec]    (choix non encore résolus)
    all_choices_for_mode(report_mode, gender_segmentation)
        → list[ChoiceSpec]    (catalogue complet, indépendant de l'état)

Interface publique (orchestration LangGraph) :
    build_methods_meta_pending_need(report_mode, gender, study_plan)
        → dict | None         (le `_pending_need` à poser, ou None)
    handle_methods_choice_response(pending, last_text, data_store, report_mode)
        → dict                (update LangGraph à retourner par le nœud)
    handle_per_tool_method_response(pending, last_text, data_store, report_mode)
        → dict                (update LangGraph pour la chaîne par-tool)

Politique : on ne propose à l'utilisateur que les choix concernant les
tools effectivement appelés dans le `report_mode` courant — pas la
liste exhaustive de tous les tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class ChoiceSpec:
    """Description d'un choix de méthode à poser à l'utilisateur."""
    tool:    str         # nom complet tool (ex: "builder.crude_rates")
    param:   str         # nom du paramètre (ex: "method")
    choices: list[str]   # valeurs autorisées (ex: ["central", "binomial", "kaplan_meier"])
    default: str         # valeur par défaut du tool
    label:   str         # libellé humain pour la question (ex: "Méthode de calcul des taux bruts")

    @property
    def context_key(self) -> str:
        """Clé sous laquelle le choix de l'utilisateur est stocké
        (ex: 'method_builder.crude_rates')."""
        return f"method_{self.tool}"


# ── Tools impliqués par mode (source unique) ──────────────────────────────
#
# On déclare ici les tools qui ont vraisemblablement >1 méthode appelables
# selon le mode. La liste reste petite et locale ; toute extension passe
# par cette table (pas de magic ailleurs).

_TOOLS_BY_MODE: dict[str, list[str]] = {
    "description": [
        # Stats descriptives uniquement — pas de méthode multi-valeurs.
    ],
    "raw_rates": [
        "builder.crude_rates",
    ],
    "full_report": [
        "builder.crude_rates",
        "builder.smoothing",
        "builder.validation",
    ],
}

# Tools utiles uniquement en by_sex (Cox compare H/F)
_TOOLS_BY_SEX_ONLY: list[str] = ["builder.cox_regression"]


# ── Libellés humains (i18n FR) ────────────────────────────────────────────

_LABELS: dict[str, str] = {
    "builder.crude_rates":    "Méthode de calcul des taux bruts",
    "builder.smoothing":      "Méthode de lissage des taux bruts",
    "builder.validation":     "Méthode de validation observé vs prédit",
    "builder.cox_regression": "Test de comparaison H/F (modèle de Cox)",
}


# ── Découverte des params multi-valeurs depuis le catalogue ───────────────

# Noms de params qu'on traite comme "méthode" (proposés à l'utilisateur).
# On exclut les paramètres de tuning numérique (`d`, `alpha`, `lambda_wh`),
# les sélecteurs de colonne (`qx_col`), ou les filtres simples (`sexe`).
_METHOD_PARAM_NAMES: set[str] = {"method", "function_name"}


def _params_with_choices(tool_name: str) -> list[tuple[str, list[str], str]]:
    """Retourne [(param_name, values_list, default)] pour les params du
    tool qui exposent une enum (`values: A | B | C`) ET qui sont
    sémantiquement un choix de méthode (cf. _METHOD_PARAM_NAMES).
    """
    try:
        from tools.catalogue import get_catalogue
    except Exception:
        return []
    cat = get_catalogue() or {}
    tool = cat.get("tools", {}).get(tool_name) or {}
    params = tool.get("params") or {}
    result: list[tuple[str, list[str], str]] = []
    for pname, pinfo in params.items():
        if pname not in _METHOD_PARAM_NAMES:
            continue
        if not isinstance(pinfo, dict):
            continue
        values_raw = pinfo.get("values")
        if not values_raw:
            continue
        # Le catalogue parse "central | binomial | kaplan_meier" → ["central", "binomial", "kaplan_meier"]
        if isinstance(values_raw, str):
            values = [v.strip() for v in values_raw.split("|") if v.strip()]
        elif isinstance(values_raw, list):
            values = [str(v).strip() for v in values_raw if str(v).strip()]
        else:
            continue
        if len(values) < 2:
            continue
        default = str(pinfo.get("default") or values[0])
        result.append((pname, values, default))
    return result


def all_choices_for_mode(
    report_mode: str,
    gender_segmentation: str | None = None,
) -> list[ChoiceSpec]:
    """Liste exhaustive des choix de méthode pour ce mode (sans filtrer
    selon study_plan)."""
    tools = list(_TOOLS_BY_MODE.get(report_mode) or [])
    if gender_segmentation == "by_sex":
        tools = tools + _TOOLS_BY_SEX_ONLY
    out: list[ChoiceSpec] = []
    for tool in tools:
        for pname, values, default in _params_with_choices(tool):
            out.append(ChoiceSpec(
                tool=tool, param=pname,
                choices=values, default=default,
                label=_LABELS.get(tool, tool),
            ))
    return out


def method_choices_for_mode(
    report_mode: str,
    gender_segmentation: str | None = None,
    study_plan: dict | None = None,
) -> list[ChoiceSpec]:
    """Retourne uniquement les choix non encore résolus.

    Un choix est considéré résolu si :
      - study_plan["methods"][tool] est défini, OU
      - study_plan["methods_auto"] est True (l'utilisateur a délégué).
    """
    sp = study_plan or {}
    if sp.get("methods_auto") is True:
        return []
    methods_picked = (sp.get("methods") or {})
    all_choices = all_choices_for_mode(report_mode, gender_segmentation)
    return [c for c in all_choices if c.tool not in methods_picked]


def llm_fallback_resolve_methods(
    answer: str,
    specs: list["ChoiceSpec"],
) -> dict[str, str]:
    """Fallback LLM appelé UNIQUEMENT quand le matching regex/alias a
    échoué pour tous les tools. Soumet la phrase utilisateur + la liste
    des choix autorisés par tool, et demande au modèle de retourner
    `{tool: method}` strictement contraint à la liste.

    Validation post-LLM : on rejette toute valeur hors liste et on filtre
    les valeurs vides. Retourne {} si l'appel échoue (réseau, parsing, …).
    """
    if not answer or not specs:
        return {}
    import json as _json
    import openai
    tools_block = "\n".join(
        f"  {s.tool}: [{', '.join(s.choices)}]" for s in specs
    )
    prompt = (
        "Tu reçois une phrase utilisateur en français qui exprime un choix de "
        "méthodes de calcul actuariel.\n\n"
        f"Phrase utilisateur : \"{answer}\"\n\n"
        f"Pour chacun des tools suivants, choisis UNE valeur dans sa liste "
        f"autorisée (ou la chaîne \"auto\" si l'utilisateur ne mentionne pas "
        f"ce tool) :\n{tools_block}\n\n"
        "Règles strictes :\n"
        "  - Aucune valeur en dehors de la liste autorisée.\n"
        "  - Tolère les typos phonétiques (kaplain → kaplan_meier, "
        "whitaker → whittaker, etc.).\n"
        "  - Réponds par un JSON unique de forme "
        "{\"<tool>\": \"<method ou auto>\"} pour chaque tool listé."
    )
    try:
        from agents.mortality.agents.llm_config import get_llm_config
        cfg = get_llm_config("master.method_resolution")
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=cfg.get("model", "gpt-5.4-mini"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 200),
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _json.loads(raw or "{}")
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, str] = {}
    allowed = {s.tool: set(s.choices) for s in specs}
    for tool_key, val in parsed.items():
        if not isinstance(val, str) or tool_key not in allowed:
            continue
        v = val.strip()
        if v == "auto" or v in allowed[tool_key]:
            out[tool_key] = v
    return out


def resolve_user_answer_to_method(
    answer: str,
    choices: Iterable[str],
) -> str | None:
    """Tente de matcher la réponse user à une des méthodes de la liste.
    Tolérant à la casse, accents, typos courants et extraction d'un mot
    parmi une phrase plus longue.
    Retourne None si aucun match.
    """
    if not answer:
        return None
    import re
    ans_lc = answer.strip().lower()
    choices_list = list(choices)
    # 1. Match direct (substring dans les deux sens)
    for c in choices_list:
        c_lc = c.strip().lower()
        if ans_lc == c_lc or c_lc in ans_lc or ans_lc in c_lc:
            return c
    # 2. Alias étendus + variantes typo fréquentes
    aliases = {
        "kaplan":               "kaplan_meier",
        "kaplain":              "kaplan_meier",     # typo phonétique fréquent
        "km":                   "kaplan_meier",
        "meier":                "kaplan_meier",
        "kaplan_meir":          "kaplan_meier",     # typo fréquent
        "kaplain_meier":        "kaplan_meier",
        "kaplain_meir":         "kaplan_meier",
        "kaplan-meier":         "kaplan_meier",
        "kaplanmeier":          "kaplan_meier",
        "wh":                   "whittaker",
        "whitaker":             "whittaker",        # typo
        "whittaker-henderson":  "whittaker",
        "whittaker_henderson":  "whittaker",
        "ic":                   "confidence_intervals",
        "ci":                   "confidence_intervals",
        "intervalles":          "confidence_intervals",
        "intervalle":           "confidence_intervals",
        "confiance":            "confidence_intervals",
        "confidence":           "confidence_intervals",
        "chi2":                 "chi_square",
        "chi-deux":             "chi_square",
        "chideux":              "chi_square",
        "khi2":                 "chi_square",
        "binom":                "binomial",
        "centrale":             "central",
    }
    choice_set = {c.strip().lower() for c in choices_list}
    # 3. Tokenisation : on découpe sur espaces et ponctuation, on tente match
    #    direct ou via alias sur chaque token.
    tokens = re.findall(r"[a-zà-ÿ_-]+", ans_lc)
    for tok in tokens:
        # Match direct sur le token
        if tok in choice_set:
            for c in choices_list:
                if c.strip().lower() == tok:
                    return c
        # Alias
        target = aliases.get(tok)
        if target and target in choice_set:
            for c in choices_list:
                if c.strip().lower() == target:
                    return c
    # 4. Alias appliqué à la réponse entière (cas legacy)
    target = aliases.get(ans_lc)
    if target and target in choice_set:
        for c in choices_list:
            if c.strip().lower() == target:
                return c
    return None


# ─────────────────────────────────────────────────────────────────────────
# Orchestration conversationnelle (anciennement dans master_node.py).
#
# Ces fonctions encapsulent toute la logique de désambiguation des
# méthodes : poser la méta-question, traiter la réponse (auto, préciser,
# inline-parse, fallback LLM, re-ask), enchaîner les questions par-tool.
# Elles retournent un dict d'update LangGraph que le nœud appelant
# (master_node) peut retourner tel quel.
# ─────────────────────────────────────────────────────────────────────────

_AGENT_SWITCH_EVENT = {"type": "agent_switch", "agent": "MasterAgent"}


def _msg_event(content: str) -> dict:
    return {"type": "message", "content": content}


def _master_instr(content: str):
    from langchain_core.messages import HumanMessage as _HMsg
    return _HMsg(content=content, additional_kwargs={"source": "master_synthetic"})


def _ai_question(content: str):
    from langchain_core.messages import AIMessage as _AIMsg
    return _AIMsg(content=content)


def _question_text_for_spec(spec: ChoiceSpec) -> str:
    return (
        f"{spec.label} — choisissez parmi : "
        + ", ".join(spec.choices) + f" (défaut : {spec.default})."
    )


def _build_pending_need_for_spec(spec: ChoiceSpec) -> dict:
    return {
        "context_key":   spec.context_key,
        "question":      _question_text_for_spec(spec),
        "options":       spec.choices + ["auto"],
        "default":       spec.default,
        "_method_tool":  spec.tool,
        "_method_param": spec.param,
    }


def build_methods_meta_pending_need(
    report_mode: str,
    gender: str | None,
    study_plan: dict | None,
) -> dict | None:
    """Construit le `_pending_need` à poser la 1re fois qu'on détecte des
    choix de méthodes. Retourne None si aucun choix à faire ou si la
    décision est déjà prise (methods_auto=True ou study_plan.methods rempli)."""
    sp = study_plan or {}
    if sp.get("methods_auto") is True or (sp.get("methods") or {}):
        return None
    pending = method_choices_for_mode(report_mode, gender, sp)
    if not pending:
        return None
    tool_list_fr = ", ".join(
        f"{c.label} ({'/'.join(c.choices)})" for c in pending
    )
    return {
        "context_key": "methods_choice_mode",
        "question": (
            "Plusieurs méthodes de calcul sont disponibles pour ce pipeline :\n"
            f"  • {tool_list_fr}\n"
            "Souhaitez-vous :\n"
            "  (a) préciser vous-même les méthodes ;\n"
            "  (b) laisser l'agent choisir (méthodes par défaut, recommandé) ?"
        ),
        "options": ["preciser", "auto"],
        "default": "auto",
    }


def _prepend_stages(updates: dict, data_store: dict) -> dict:
    """Injecte les stages accumulés par master_node dans les events du
    dict de retour. No-op si pas de buffer (cas tests unitaires)."""
    buffered = data_store.pop("_stage_buffer", []) or []
    if buffered:
        existing = updates.get("events") or []
        updates["events"] = list(buffered) + list(existing)
    return updates


def _route_to_builder(data_store: dict, instr_text: str, event_text: str) -> dict:
    return _prepend_stages({
        "messages":     [_master_instr(instr_text)],
        "events":       [_AGENT_SWITCH_EVENT, _msg_event(event_text)],
        "active_agent": "builder",
        "data_store":   data_store,
    }, data_store)


def _ask_user(data_store: dict, question: str) -> dict:
    return _prepend_stages({
        "messages":   [_ai_question(question)],
        "events":     [_AGENT_SWITCH_EVENT, _msg_event(question)],
        "data_store": data_store,
    }, data_store)


# Regex de détection des questions génériques (vs choix de méthode).
# Match : "?", "qu'est-ce", "rappelle", "explique", "c'est quoi",
# "comment", "pourquoi", "différence entre", "donne moi". Insensible casse.
import re as _re
_QUESTION_PATTERN = _re.compile(
    r"\?|qu['e]?st[\s-]?ce|rappelle|expliqu|c'?est quoi|comment\b|pourquoi"
    r"|diff[ée]rence|donne[\s-]moi|d[ée]tail|c'?est quoi|aide|info\b",
    _re.IGNORECASE,
)


def _is_meta_question(text: str) -> bool:
    """Détecte si le message est une QUESTION (besoin d'info) plutôt qu'un
    CHOIX de méthode. Heuristique : présence d'un signal interrogatif fort
    (?, qu'est-ce, explique, rappelle, comment, pourquoi, différence,
    donne moi, c'est quoi). Le fait que la phrase contienne un nom de
    méthode n'invalide pas — l'utilisateur peut demander des explications
    SUR la méthode (ex: 'explique-moi kaplan' est une question légitime)."""
    if not text:
        return False
    return bool(_QUESTION_PATTERN.search(text))


def _answer_meta_question_keeping_pending(
    pending: dict, last_text: str, data_store: dict,
) -> dict:
    """L'utilisateur a posé une question pendant un pending_need méthode.
    On appelle search_doctrine pour répondre, et on RE-POSE le pending
    pour que la conversation reprenne après."""
    # Stage tracking pour l'UI "internal agent"
    if "_stage_buffer" in data_store and isinstance(data_store["_stage_buffer"], list):
        data_store["_stage_buffer"].append({
            "type":  "master_stage",
            "stage": "0.c-q",
            "label": "Échappe-question pendant pending méthode (RAG doctrine)",
        })
    from tools.conversation.search_doctrine import run as _search
    res = _search(None, {"query": last_text, "k": 3})

    if "erreur" in res or not res.get("results"):
        # Fallback : juste re-poser sans contexte enrichi
        hint = (
            f"Je n'ai pas trouvé de doc spécifique sur '{last_text[:80]}'. "
            f"{pending.get('question', '')}"
        )
        return _ask_user(data_store, hint)

    # Formuler une réponse courte via les chunks (le LLM nano ne tourne
    # PAS ici — on fait du templating pour rester rapide et déterministe).
    lines = [f"Voici ce que dit la doctrine sur votre question :\n"]
    for r in res["results"][:3]:
        title = f"{r['doc_id']}.{r['section_id']} — {r['section_title']}"
        excerpt = r["text"][:400].rsplit(" ", 1)[0] + "…"
        lines.append(f"\n**{title}**\n{excerpt}\n")
    lines.append(
        f"\n---\nReprenons : {pending.get('question', '')}"
    )
    return _ask_user(data_store, "\n".join(lines))


def handle_methods_choice_response(
    pending: dict,
    last_text: str,
    data_store: dict,
    *,
    report_mode: str,
) -> dict:
    """Traite la réponse user à la méta-question `methods_choice_mode`.
    Quatre branches :
      - QUESTION → search_doctrine + ré-affichage du pending (sans le consommer)
      - "auto" / "b)" / "laisse"  → methods_auto=True, route Builder.
      - "préciser" / "a)"         → enchaîne la 1re question per-tool.
      - réponse libre             → inline-parse (regex) puis fallback LLM,
                                     puis re-ask si rien n'a marché.
    """
    # Échappe-question : l'utilisateur veut info avant de choisir
    if _is_meta_question(last_text):
        return _answer_meta_question_keeping_pending(pending, last_text, data_store)

    ans_lc = (last_text or "").strip().lower()
    sp = data_store.setdefault("study_plan", {})
    data_store["_methods_question_done"] = True

    # Branche "auto"
    if any(w in ans_lc for w in ("auto", "agent choisi", "agent choisis",
                                   "laisse", "défaut", "defaut", "b)")):
        sp["methods_auto"] = True
        data_store.pop("_pending_need", None)
        return _route_to_builder(
            data_store,
            "[Master] Choix méthodes : automatique (defaults du pipeline).",
            "Choix automatique des méthodes par l'agent.",
        )

    gender = sp.get("gender_segmentation") or data_store.get("gender_segmentation")

    # Branche "préciser" → enchaîne la 1re question per-tool
    if any(w in ans_lc for w in ("preciser", "préciser", "moi-meme",
                                   "moi-même", "a)", "manuel", "choisir")):
        remaining = method_choices_for_mode(report_mode, gender, sp)
        if not remaining:
            data_store.pop("_pending_need", None)
            return {
                "messages":     [],
                "events":       [_AGENT_SWITCH_EVENT,
                                 _msg_event("Aucune méthode à préciser — on continue.")],
                "active_agent": "builder",
                "data_store":   data_store,
            }
        data_store["_pending_need"] = _build_pending_need_for_spec(remaining[0])
        return _ask_user(data_store, data_store["_pending_need"]["question"])

    # Branche réponse libre : inline-parse (regex/alias)
    specs = all_choices_for_mode(report_mode, gender)
    methods_picked = sp.setdefault("methods", {})
    matched_any = False
    for spec in specs:
        hit = resolve_user_answer_to_method(ans_lc, spec.choices)
        if hit:
            methods_picked[spec.tool] = hit
            matched_any = True
    if matched_any:
        for spec in specs:
            methods_picked.setdefault(spec.tool, "auto")
        data_store.pop("_pending_need", None)
        return _route_to_builder(
            data_store,
            f"[Master] Méthodes choisies (parsing inline) : {methods_picked}.",
            f"Méthodes enregistrées : {methods_picked}.",
        )

    # Fallback LLM
    llm_resolved = llm_fallback_resolve_methods(last_text, specs)
    if llm_resolved:
        for tool_key, val in llm_resolved.items():
            methods_picked[tool_key] = val
        for spec in specs:
            methods_picked.setdefault(spec.tool, "auto")
        data_store.pop("_pending_need", None)
        return _route_to_builder(
            data_store,
            f"[Master] Méthodes choisies (fallback LLM) : {methods_picked}.",
            f"Méthodes enregistrées : {methods_picked}.",
        )

    # Re-ask
    return _ask_user(data_store, (
        f"Je n'ai pas compris votre choix '{last_text}'. Répondez par "
        f"'auto' (l'agent choisit) ou 'préciser' (vous indiquez les méthodes), "
        f"ou indiquez directement les méthodes (ex: 'kaplan_meier, whittaker, "
        f"confidence_intervals')."
    ))


def handle_per_tool_method_response(
    pending: dict,
    last_text: str,
    data_store: dict,
    *,
    report_mode: str,
) -> dict:
    """Traite la réponse user à une question per-tool (context_key
    démarre par 'method_'). Enchaîne la question suivante si plusieurs
    tools restent à choisir, sinon route vers Builder.
    Échappe-question : si le user pose une question, on appelle
    search_doctrine et on re-pose la question méthode."""
    # Échappe-question : info avant choix
    if _is_meta_question(last_text):
        return _answer_meta_question_keeping_pending(pending, last_text, data_store)

    tool_name = pending.get("_method_tool", "")
    choices = pending.get("options") or []
    ans = (last_text or "").strip().lower()
    sp = data_store.setdefault("study_plan", {})
    methods_picked = sp.setdefault("methods", {})

    if ans in ("auto", "default", "défaut", "defaut") or "auto" in ans:
        methods_picked[tool_name] = "auto"
    else:
        resolved = resolve_user_answer_to_method(
            ans, [c for c in choices if c != "auto"],
        )
        if resolved is None:
            options_str = ", ".join(choices)
            return _ask_user(data_store, (
                f"Je n'ai pas reconnu '{last_text}'. "
                f"Pour {pending.get('question', 'cette méthode')}, "
                f"choisissez : {options_str}."
            ))
        methods_picked[tool_name] = resolved

    data_store.pop("_pending_need", None)
    gender = sp.get("gender_segmentation") or data_store.get("gender_segmentation")
    remaining = method_choices_for_mode(report_mode, gender, sp)
    if remaining:
        data_store["_pending_need"] = _build_pending_need_for_spec(remaining[0])
        return _ask_user(data_store, data_store["_pending_need"]["question"])

    return _route_to_builder(
        data_store,
        f"[Master] Méthodes choisies : {methods_picked}.",
        f"Méthodes enregistrées : {methods_picked}.",
    )
