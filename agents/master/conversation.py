"""
agents/master/conversation.py
Logique conversationnelle du Master — extraite de master_node.py:937-968.

Quand classify_intent → kind="question", on entre ici. Au lieu d'une simple
génération de texte, le LLM a accès à un set restreint de tools d'inspection
et d'exploration : data_inspect, plot_basic, eval_pandas, et les tools
statistical_analysis.* (lecture seule).

Scope strict :
  - Le Builder ne voit JAMAIS ces tools (cf. BUILDER_TOOLS dans builder_node).
  - Les tools "calcul actuariel" (builder.*, build_pdf, aggregation,
    preprocessing) ne sont JAMAIS exposés ici.

L'utilisateur ne tape jamais de code Python — le LLM génère soit un tool
call structuré, soit une expression pandas validée par AST.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, ToolMessage

if TYPE_CHECKING:
    pass


# Whitelist des tools accessibles en mode conversationnel.
# Le mécanisme tool-calling reste celui d'OpenAI ; on filtre simplement la
# liste passée à l'API.
CONVERSATIONAL_TOOLS: set[str] = {
    "statistical_analysis",   # data_quality, age_distribution, time_series,
                              # segmentation, portfolio_summary, descriptive_report
    "conversation",           # data_inspect, plot_basic, eval_pandas
}

# Borne de la boucle tool-calling — au-delà on retourne le message courant.
_MAX_TOOL_ITERATIONS = 5


def _filtered_openai_tools() -> list[dict]:
    """Retourne le sous-ensemble de get_openai_tools() restreint au scope
    conversationnel."""
    from tools.tool_registry import get_openai_tools
    return [t for t in get_openai_tools()
            if t.get("function", {}).get("name") in CONVERSATIONAL_TOOLS]


def _load_df(dataset_ref: str | None, data_store: dict):
    """Charge le DataFrame en préférant le Parquet normalisé (cf. étape 3
    du pipeline). Retourne None si pas de dataset."""
    if not dataset_ref and not (data_store or {}).get("dataset_ref_normalized"):
        return None
    try:
        from session.dataset_store import DatasetStore
        return DatasetStore.load_preferring_normalized(data_store or {}, dataset_ref)
    except Exception:
        return None


def respond_conversationally(
    messages_list: list,
    data_store: dict,
    dataset_ref: str | None,
) -> dict:
    """Branche conversationnelle du Master.

    Boucle tool-calling avec CONVERSATIONAL_TOOLS uniquement :
      1. Envoie le prompt + historique + tools au LLM (gpt-5.4-nano).
      2. Si le LLM décide d'appeler un tool → exécute via call_tool() →
         renvoie le ToolMessage → continue.
      3. Sinon → réponse finale, fin de boucle.

    Retourne un update LangGraph (messages, events, data_store) tel que
    master_node peut le retourner directement.
    """
    import openai
    from agents.mortality.agents.master_node import _build_system_prompt, _augment_with_data_store
    from agents.mortality.agents.mortality_node import (
        _to_openai_dict, _from_openai_response, sanitize_openai_messages,
    )
    from agents.mortality.agents._utils import call_with_retry
    from agents.mortality.agents.llm_config import get_llm_config
    from tools.tool_registry import call_tool

    system_prompt = _augment_with_data_store(
        _build_system_prompt(), data_store, dataset_ref,
    )
    # Ajouter une note explicite sur les tools dispo pour orienter le LLM
    system_prompt += (
        "\n\n## ⚠️ OVERRIDE — Mode conversationnel actif\n"
        "Tu es actuellement en MODE CONVERSATION, pas en mode routing. "
        "Les règles 'Tu ne fais pas de calculs' / 'Tu ne choisis pas les "
        "méthodes' du contrat comportemental NE S'APPLIQUENT PAS ICI. "
        "En conversation, tu DOIS appeler les tools ci-dessous chaque fois "
        "qu'ils peuvent aider à répondre à l'utilisateur. Ne jamais inventer "
        "que tu manques de 'contexte' — appelle d'abord le tool, vois ce "
        "qu'il retourne, puis réponds.\n\n"
        "## Exemple obligatoire\n\n"
        "User : « C'est quoi le lissage Whittaker-Henderson ? »\n"
        "Toi : appelle IMMÉDIATEMENT `conversation.search_doctrine` avec\n"
        "      `query=\"lissage Whittaker-Henderson\"` puis reformule la\n"
        "      réponse en citant la source (ex: « Selon D03.02 - "
        "Whittaker-Henderson 1D... »). NE PAS répondre de mémoire, NE PAS "
        "inventer d'excuse sur un manque de contexte.\n\n"
        "## Outils d'exploration disponibles\n"
        "Tu PEUX appeler ces tools pour répondre :\n"
        "  - `conversation.describe_capabilities` : liste structurée de ce "
        "que le système sait faire + inputs nécessaires + rapports producibles. "
        "**APPELER systématiquement** quand l'utilisateur demande "
        "'que sais-tu faire ?', 'qu'as-tu besoin de moi ?', "
        "'comment ça marche ?', 'quels rapports peux-tu produire ?'. "
        "Reformuler le JSON retourné en langage naturel — ne pas le balancer brut.\n"
        "  - `conversation.search_doctrine` : RAG sur 142 chunks actuariels FR "
        "(préparation données, estimateurs taux bruts, lissage Whittaker-Henderson, "
        "validation chi²/runs/SMR, fermeture grands âges, tables prospectives "
        "Lee-Carter/Brouhns/Cairns, A132-18, BCAC, arrêtés, certification IA, "
        "Solvabilité 2, IFRS 17, tables TH/TF/TGH/TGF/TPRV). "
        "**APPELER systématiquement** quand l'utilisateur pose une QUESTION "
        "méthodologique ou réglementaire (ex: 'c'est quoi Whittaker-Henderson ?', "
        "'explique-moi le test du chi-2', 'quelle obligation A132-18 ?', "
        "'différence table périodique vs prospective ?', 'comment fermer les "
        "grands âges ?'). RÈGLES STRICTES de réponse : (a) reformule en "
        "langage naturel, ne balance pas le texte brut ; (b) CITE TOUJOURS les "
        "sources sous la forme 'D03.02 - Whittaker-Henderson 1D' ; (c) ne JAMAIS "
        "inventer si aucun chunk ne couvre la question — dire honnêtement "
        "'le corpus n'inclut pas ce point' ; "
        "(d) **FORMULES : ne JAMAIS copier le LaTeX brut**. Le canvas chat "
        "ne rend PAS LaTeX. Convertis systématiquement vers Unicode math :\n"
        "    \\sum → Σ, \\Delta → Δ, \\tilde{q} → q̃, \\hat{q} → q̂,\n"
        "    q_x → q_x (souscript visible),\n"
        "    \\frac{a}{b} → a/b (ou (a)/(b) si ambigu),\n"
        "    x^2 → x², x^{-1} → x⁻¹, x^z → xᶻ ou x^(z),\n"
        "    \\mathrm{diag}(w_i) → diag(w_i),\n"
        "    Présenter une formule complète indentée 4 espaces (style code "
        "plain text), JAMAIS encadrée par $…$, \\[…\\], \\(…\\) ou [ … ].\n"
        "    Exemple :\n"
        "        M(q̃) = h Σᵢ wᵢ (qᵢ − q̃ᵢ)² + Σᵢ (Δᶻ q̃ᵢ)²\n"
        "    où q̃ = (W + h⁻¹ KᵀK)⁻¹ W q\n"
        "  - `conversation.data_inspect` : colonnes, head, describe, "
        "value_counts, date_range, shape.\n"
        "  - `conversation.plot_basic` : histogram, bar, scatter, time_series → PNG.\n"
        "  - `conversation.eval_pandas` : expression Python sandboxée "
        "(df, pd, np, plt, sns, stats, ll/lifelines disponibles).\n"
        "  - `conversation.apply_normalization` : déclenche la normalisation "
        "complète du fichier (mapping + dates + sentinelles) à la demande user.\n"
        "  - `statistical_analysis.*` : tools descriptifs déjà existants.\n"
        "Tu NE PEUX PAS lancer de calculs actuariels (builder, build_pdf, aggregation, "
        "preprocessing) — pour ça, l'utilisateur doit explicitement demander un rapport "
        "ou des calculs et ce sera routé vers le Builder.\n"
        "Quand tu utilises `eval_pandas`, écris des expressions Python pures "
        "(pas d'import, pas d'assignation). Variables : `df`, `pd`, `np`, "
        "`plt`, `sns`, `stats`, `ll`, `datetime`."
    )

    cfg = get_llm_config("master.conversation")
    client = openai.OpenAI()

    raw_msgs = messages_list[-20:]
    messages = [{"role": "system", "content": system_prompt}]
    messages += [_to_openai_dict(m) for m in raw_msgs]
    messages = sanitize_openai_messages(messages)

    tools_schema = _filtered_openai_tools()
    df = _load_df(dataset_ref, data_store)

    new_events: list[dict] = [{"type": "agent_switch", "agent": "MasterAgent"}]
    new_lc_messages: list = []
    plots_produced: list[str] = []

    for iteration in range(_MAX_TOOL_ITERATIONS):
        try:
            response = call_with_retry(
                client,
                model=cfg["model"],
                messages=messages,
                tools=tools_schema if tools_schema else None,
                tool_choice="auto" if tools_schema else None,
                max_tokens=cfg.get("max_tokens", 1500),
                temperature=cfg.get("temperature", 0.3),
            )
        except Exception as exc:
            new_events.append({"type": "error", "message": str(exc)})
            return {"messages": new_lc_messages, "events": new_events,
                    "data_store": data_store}

        choice = response.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None) or []

        # Toujours convertir le message LLM en LangChain pour traçabilité
        lc_msg = _from_openai_response(msg)
        new_lc_messages.append(lc_msg)
        # Et l'injecter dans la conversation pour le tour suivant
        messages.append({
            "role":       "assistant",
            "content":    msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in tool_calls
            ] if tool_calls else None,
        })

        if not tool_calls:
            # Réponse finale
            content = msg.content or ""
            if content:
                new_events.append({"type": "message", "content": content})
            for path in plots_produced:
                new_events.append({"type": "image", "path": path})
            new_events.append({"type": "done"})
            return {"messages": new_lc_messages, "events": new_events,
                    "data_store": data_store}

        # Exécuter chaque tool_call
        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = {}
            function_name = fn_args.get("function_name", "")
            params = fn_args.get("params", {})

            new_events.append({
                "type":          "tool_call",
                "tool":          fn_name,
                "function_name": function_name,
                "params":        params,
                "tool_call_id":  tc.id,
            })

            if fn_name not in CONVERSATIONAL_TOOLS:
                # Le LLM tente un tool hors scope (ne devrait pas arriver avec
                # la whitelist côté API, mais ceinture+bretelle)
                result = {"erreur": f"tool '{fn_name}' hors scope conversationnel"}
            else:
                result = call_tool(
                    tool_name=fn_name,
                    function_name=function_name,
                    params=params,
                    df=df,
                    data=data_store,
                )

            # Collecter les plots PNG du résultat
            for key in ("png_path", "plots"):
                v = result.get(key) if isinstance(result, dict) else None
                if isinstance(v, str):
                    plots_produced.append(v)
                elif isinstance(v, list):
                    plots_produced.extend(p for p in v if isinstance(p, str))

            new_events.append({
                "type":          "tool_result",
                "tool":          fn_name,
                "function_name": function_name,
                "result":        result,
                "tool_call_id":  tc.id,
            })

            # ToolMessage pour LLM
            tool_msg = ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str)[:6000],
                tool_call_id=tc.id,
            )
            new_lc_messages.append(tool_msg)
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      tool_msg.content,
            })

    # Bornage atteint — retourner ce qu'on a
    new_events.append({
        "type":    "message",
        "content": f"[Master] Limite de {_MAX_TOOL_ITERATIONS} tool calls atteinte.",
    })
    new_events.append({"type": "done"})
    return {"messages": new_lc_messages, "events": new_events,
            "data_store": data_store}
