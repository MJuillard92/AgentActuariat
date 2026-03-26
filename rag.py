"""
rag.py
RAG + Tool Use sur les résultats du pipeline actuariel.

Deux modes :
  1. answer_with_rag(question, steps, ...)   – RAG classique avec cache d'embeddings
  2. answer_with_tools(question, steps, ...) – RAG + tool use (exec Python, accès DataFrames)

Pipeline RAG (Retrieval-Augmented Generation) :
  a. Lors de la fin de l'analyse → precompute_index() calcule et met en cache
     les embeddings de TOUS les chunks (sorties des étapes).
  b. À chaque question utilisateur :
       - on embède seulement la question (1 appel API léger),
       - on calcule la similarité cosinus entre la question et les chunks en cache,
       - on sélectionne les top_k chunks les plus proches,
       - on injecte ces chunks + la question dans le prompt du LLM.

Optimisations mémoire/coût :
  - precompute_index() est appelé une seule fois après l'analyse : les embeddings
    des chunks ne sont jamais recalculés entre les questions.
  - _dynamic_window() ajuste top_k et max_tokens selon le budget de la fenêtre
    contextuelle : on maximise le contexte sans dépasser la limite du modèle.
  - answer_with_tools() ajoute une boucle de tool-use (max 6 tours) permettant
    au LLM de recalculer des indicateurs manquants via execute_python().

Dispatch des outils (answer_with_tools) :
  - execute_python    → _exec_python()         : exécute du code dans le kernel
  - list_available_data → _list_available_data() : liste les DataFrames disponibles
  - get_dataframe_info → _get_dataframe_info()  : affiche le détail d'un DataFrame
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import traceback
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

import config

load_dotenv()

_EMBED_MODEL = "text-embedding-3-small"
_MAX_CHUNK_CHARS = 1500   # Taille maximale d'un chunk : suffisant pour capturer
                           # un tableau ou un paragraphe de log, assez petit pour
                           # rentrer plusieurs chunks dans la fenêtre du LLM.

# ── Fenêtre contextuelle dynamique ───────────────────────────────────────────
_MODEL_CONTEXT_TOKENS = 128_000   # gpt-4o-mini / gpt-4o
_RESPONSE_RESERVE_TOKENS = 4_096  # tokens réservés pour la réponse du modèle
_CHARS_PER_TOKEN = 4              # approximation grossière : 1 token ≈ 4 caractères
                                   # (valable pour le français, légèrement sous-estimé)


def _dynamic_window(
    chunks: list[dict],
    question: str,
    summary: str = "",
    system_prompt: str = "",
) -> tuple[int, int]:
    """Calcule dynamiquement top_k et max_tokens selon le budget contextuel du modèle.

    Maximise le nombre de chunks RAG injectés sans dépasser la fenêtre du modèle,
    puis réserve le solde pour la réponse.

    Returns:
        (top_k, max_tokens)
    """
    n = len(chunks)
    if n == 0:
        return 0, _RESPONSE_RESERVE_TOKENS

    # Tokens consommés par les éléments fixes (hors chunks)
    fixed_tokens = (
        len(system_prompt) // _CHARS_PER_TOKEN
        + len(question) // _CHARS_PER_TOKEN
        + len(summary) // _CHARS_PER_TOKEN
        + 300  # overhead : délimiteurs, rôles, formatage
    )

    # Budget disponible pour les chunks RAG
    chunk_budget = _MODEL_CONTEXT_TOKENS - _RESPONSE_RESERVE_TOKENS - fixed_tokens
    tokens_per_chunk = (_MAX_CHUNK_CHARS // _CHARS_PER_TOKEN) + 30  # chunk + délimiteurs

    # Nombre de chunks injectables sans dépasser le budget
    top_k = max(1, min(n, chunk_budget // tokens_per_chunk))

    # max_tokens pour la réponse = solde après system + question + summary + chunks
    used_tokens = fixed_tokens + top_k * tokens_per_chunk
    remaining = _MODEL_CONTEXT_TOKENS - used_tokens
    max_tokens = max(512, min(_RESPONSE_RESERVE_TOKENS * 2, remaining))

    return top_k, max_tokens

RAG_SYSTEM_PROMPT = (
    "Tu es un expert actuariel pédagogue spécialisé en tables de mortalité d'expérience. "
    "Tu réponds aux questions en te basant UNIQUEMENT sur les extraits fournis dans le contexte. "
    "Tu cites les valeurs numériques précises présentes dans les logs (SMR, qx, E_x, D_x, etc.). "
    "Tu expliques clairement les concepts actuariels sans jargon excessif. "
    "Si l'information n'est pas dans le contexte, tu le signales honnêtement. "
    "Réponds en français."
)

RAG_TOOLS_SYSTEM_PROMPT = (
    "Tu es un expert actuariel disposant d'un accès complet aux données et résultats de l'analyse. "
    "Tu peux exécuter du code Python dans le kernel de l'analyse.\n\n"
    "RÈGLES STRICTES :\n"
    "1. Commence TOUJOURS par appeler list_available_data() pour voir ce qui est disponible.\n"
    "2. N'invente JAMAIS de noms de colonnes ou de variables — vérifie avec get_dataframe_info().\n"
    "3. Pour afficher des résultats, utilise TOUJOURS print() dans execute_python.\n"
    "4. Si une variable n'existe pas, dis-le clairement et propose une alternative.\n\n"
    "DONNÉES DISPONIBLES TYPIQUEMENT :\n"
    "- df : DataFrame brut chargé (avant nettoyage)\n"
    "- df_clean : DataFrame après nettoyage\n"
    "- data_prep.df_removed : lignes supprimées avec colonne 'removal_reason'\n"
    "- df_exposure, df_qx, df_smooth : tables calculées en aval\n"
    "- Modules actuariels : data_prep, exposure, crude_rates, smoothing, diagnostics, validation, benchmarking, visualization\n\n"
    "POUR LES LIGNES SUPPRIMÉES :\n"
    "  execute_python('print(data_prep.df_removed.to_string())')\n"
    "  execute_python('print(data_prep.df_removed.groupby(\"removal_reason\").size())')\n\n"
    "Réponds en français avec des valeurs numériques précises."
)

# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY manquante dans .env")
    return OpenAI(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Indexation
# ─────────────────────────────────────────────────────────────────────────────

def build_index(steps: list[dict]) -> list[dict]:
    """Construit la liste de chunks depuis les sorties d'étapes.

    Chaque étape devient un chunk (texte + label). Les sorties vides sont ignorées
    pour ne pas polluer l'index. Les sorties trop longues sont tronquées en deux
    moitiés (début + fin) pour conserver l'en-tête et les dernières lignes.
    """
    chunks = []
    for step in steps:
        label = step.get("label") or step.get("description") or "Étape"
        output = (step.get("output") or "").strip()
        if not output:
            continue
        if len(output) > _MAX_CHUNK_CHARS:
            half = _MAX_CHUNK_CHARS // 2
            output = output[:half] + "\n[…]\n" + output[-half:]
        chunks.append({"text": f"[{label}]\n{output}", "label": label})
    return chunks


def build_source_chunks(
    notebooks_dir: str | None = None,
    root_files: list[str] | None = None,
    max_chars_per_chunk: int = _MAX_CHUNK_CHARS,
) -> list[dict]:
    """Construit des chunks RAG depuis le code source des modules actuariels (.py).

    Chaque fonction Python devient un chunk indépendant, identifiée par son
    nom (``def xxx``). Les fichiers trop courts (< 100 caractères) sont ignorés.

    Le RAG peut ainsi répondre à des questions comme :
      «Quelle formule utilise smooth_whittaker ?»
      «Quels paramètres accepte diagnose_credibility ?»

    Args:
        notebooks_dir: Répertoire contenant les modules .py (défaut : ./notebooks).
        root_files:    Liste des fichiers .py racine à inclure
                       (défaut : actuarial_params.py, smoothing_selector.py).
        max_chars_per_chunk: Taille maximale d'un chunk (les fonctions plus longues
                             sont tronquées).

    Returns:
        list[dict] : chaque dict a les clés ``text`` et ``label``.
    """
    import re
    from pathlib import Path as _Path

    _root = _Path(__file__).parent
    if notebooks_dir is None:
        nb_dir = _root / "notebooks"
    else:
        nb_dir = _Path(notebooks_dir)
    if root_files is None:
        root_files = ["actuarial_params.py", "smoothing_selector.py"]

    py_files: list[_Path] = sorted(nb_dir.glob("[0-9][0-9]_*.py"))
    py_files += [_root / f for f in root_files if (_root / f).exists()]

    chunks: list[dict] = []
    _func_re = re.compile(r"^(def |class )", re.MULTILINE)

    for fpath in py_files:
        try:
            src = fpath.read_text(encoding="utf-8")
        except OSError:
            continue
        if len(src) < 100:
            continue

        # Découper par définition de fonction/classe
        matches = list(_func_re.finditer(src))
        if not matches:
            # Fichier sans fonctions (ex: actuarial_params.py) → un seul chunk
            text = f"[Source: {fpath.name}]\n{src[:max_chars_per_chunk]}"
            chunks.append({"text": text, "label": f"source:{fpath.name}"})
            continue

        # Un chunk par bloc fonction/classe
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(src)
            block = src[start:end].strip()
            if len(block) < 20:
                continue
            if len(block) > max_chars_per_chunk:
                half = max_chars_per_chunk // 2
                block = block[:half] + "\n[…]\n" + block[-half:]
            # Extraire le nom de la fonction pour le label
            name_match = re.match(r"(?:def|class)\s+(\w+)", block)
            fn_name = name_match.group(1) if name_match else f"block_{i}"
            label = f"source:{fpath.stem}.{fn_name}"
            chunks.append({"text": f"[{label}]\n{block}", "label": label})

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings + cosinus
# ─────────────────────────────────────────────────────────────────────────────

def _embed(texts: list[str], client: OpenAI) -> np.ndarray:
    response = client.embeddings.create(model=_EMBED_MODEL, input=texts)
    return np.array([e.embedding for e in response.data], dtype=np.float32)


def _cosine_scores(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10
    return (doc_vecs / norms) @ q


# ─────────────────────────────────────────────────────────────────────────────
# Pré-calcul de l'index (cache dans ActuaryState)
# ─────────────────────────────────────────────────────────────────────────────

def precompute_index(steps: list[dict], state=None) -> None:
    """Pré-calcule les embeddings de tous les chunks une fois après l'analyse.

    Appeler dès que l'analyse est terminée pour que les premières questions
    soient instantanées (seule la query sera embedée à chaque appel).
    """
    chunks = build_index(steps)
    if not chunks:
        return
    try:
        client = _get_client()
        doc_vecs = _embed([c["text"] for c in chunks], client)
        if state is not None:
            state.set_embed_cache(chunks, doc_vecs)
    except Exception as exc:
        print(f"[RAG] Erreur pré-calcul embeddings : {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval commun
# ─────────────────────────────────────────────────────────────────────────────

def _retrieve(question: str, chunks: list[dict], client: OpenAI, top_k: int,
              state=None) -> list[str]:
    """Retourne les top_k textes les plus proches de la question."""
    # Utiliser le cache si disponible
    doc_vecs = None
    if state is not None:
        cached = state.get_embed_cache(chunks)
        if cached is not None:
            chunks, doc_vecs = cached

    if doc_vecs is None:
        doc_vecs = _embed([c["text"] for c in chunks], client)
        if state is not None:
            state.set_embed_cache(chunks, doc_vecs)

    query_vec = _embed([question], client)[0]
    scores = _cosine_scores(query_vec, doc_vecs)
    top_k_actual = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:top_k_actual]
    return [chunks[i]["text"] for i in top_indices]


# ─────────────────────────────────────────────────────────────────────────────
# RAG classique
# ─────────────────────────────────────────────────────────────────────────────

def answer_with_rag(
    question: str,
    steps: list[dict],
    top_k: int | None = None,
    summary: str = "",
    system_prompt: str = None,
    conversation_history: list[dict] | None = None,
    state=None,
) -> str:
    """RAG classique avec cache d'embeddings et fenêtre contextuelle dynamique."""
    if not steps:
        return "Aucun résultat disponible. Lancez d'abord une analyse."
    chunks = build_index(steps)
    if not chunks:
        return "Les étapes n'ont produit aucune sortie textuelle."

    try:
        client = _get_client()
        sys_prompt = system_prompt or RAG_SYSTEM_PROMPT

        # Fenêtre dynamique : top_k et max_tokens calculés selon le budget contextuel
        effective_top_k, max_tokens_resp = _dynamic_window(
            chunks, question, summary, sys_prompt
        )
        if top_k is not None:
            effective_top_k = min(top_k, len(chunks))

        context_parts = _retrieve(question, chunks, client, effective_top_k, state)
        if summary and summary.strip():
            context_parts.append(f"[Synthèse globale]\n{summary.strip()}")
        if state is not None:
            ns = state.summary()
            if "Aucun état" not in ns:
                context_parts.append(f"[Namespace disponible]\n{ns}")
        context = "\n\n---\n\n".join(context_parts)

        user_content = (
            f"Extraits pertinents des étapes de l'analyse :\n\n{context}"
            f"\n\n---\n\nQuestion : {question}"
        )
        messages = [{"role": "system", "content": sys_prompt}]
        for msg in (conversation_history or [])[:-1]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=config.FORMATTER_MODEL,
            messages=messages,
            max_tokens=max_tokens_resp,
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()

    except Exception as exc:
        return f"Erreur RAG : {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool use — définitions
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Exécute du code Python dans le namespace complet de l'analyse. "
                "Accès aux DataFrames (df, df_clean, df_exposure, df_qx, df_smooth, …), "
                "aux modules actuariels (data_prep, exposure, crude_rates, smoothing, "
                "diagnostics, validation, benchmarking, visualization), "
                "à pandas, numpy, matplotlib. "
                "Utilise print() pour afficher les résultats. "
                "Les graphiques matplotlib sont automatiquement capturés."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Code Python valide à exécuter.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_data",
            "description": (
                "Liste tous les objets disponibles dans le namespace de l'analyse : "
                "DataFrames, Series, arrays numpy, variables scalaires, modules."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dataframe_info",
            "description": (
                "Retourne les infos détaillées d'un DataFrame ou Series : "
                "forme, colonnes, statistiques descriptives, premières lignes, valeurs manquantes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nom exact de la variable dans le namespace.",
                    },
                    "n_rows": {
                        "type": "integer",
                        "description": "Nombre de lignes à afficher (défaut 5).",
                    },
                },
                "required": ["name"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Implémentation des outils
# ─────────────────────────────────────────────────────────────────────────────

def _capture_rag_figures(exec_ns: dict) -> list[bytes]:
    """Capture les figures matplotlib ouvertes dans le namespace."""
    plt = exec_ns.get("plt")
    if plt is None:
        return []
    figs = []
    for fn in plt.get_fignums():
        fig = plt.figure(fn)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        figs.append(buf.read())
    plt.close("all")
    return figs


def _exec_python(code: str, exec_ns: dict) -> tuple[str, list[bytes]]:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, exec_ns)  # noqa: S102
        out = buf.getvalue().strip()
        text = out if out else "✓ Exécuté sans sortie"
    except Exception:
        return f"❌ Erreur :\n{traceback.format_exc()}", []
    figures = _capture_rag_figures(exec_ns)
    if figures:
        text += f"\n[{len(figures)} graphique(s) généré(s)]"
    return text, figures


def _list_available_data(exec_ns: dict, state=None) -> str:
    if state is not None:
        return state.summary()
    import pandas as pd
    import numpy as np
    lines = ["=== Objets disponibles ==="]
    for name, val in exec_ns.items():
        if name.startswith("_") or callable(val) or isinstance(val, type):
            continue
        if isinstance(val, pd.DataFrame):
            lines.append(
                f"  • {name}: DataFrame {val.shape[0]}×{val.shape[1]}"
                f" — colonnes: {list(val.columns[:6])}"
            )
        elif isinstance(val, pd.Series):
            lines.append(f"  • {name}: Series ({len(val)})")
        elif isinstance(val, np.ndarray):
            lines.append(f"  • {name}: array {val.shape}")
        elif isinstance(val, (int, float, bool)):
            lines.append(f"  • {name} = {val!r}")
        elif isinstance(val, str) and len(val) < 100:
            lines.append(f"  • {name} = {val!r}")
    return "\n".join(lines) if len(lines) > 1 else "Aucune donnée disponible."


def _get_dataframe_info(name: str, exec_ns: dict, n_rows: int = 5) -> str:
    import pandas as pd
    val = exec_ns.get(name)
    if val is None:
        return f"'{name}' introuvable dans le namespace."
    if not isinstance(val, (pd.DataFrame, pd.Series)):
        return f"'{name}' est de type {type(val).__name__}, pas un DataFrame."
    buf = io.StringIO()
    if isinstance(val, pd.DataFrame):
        buf.write(f"Shape : {val.shape}\n")
        buf.write(f"Colonnes : {list(val.columns)}\n\n")
        buf.write(f"Premières lignes :\n{val.head(n_rows).to_string()}\n\n")
        buf.write(f"Statistiques :\n{val.describe().to_string()}\n")
        nulls = val.isnull().sum()
        if nulls.any():
            buf.write(f"\nValeurs manquantes :\n{nulls[nulls > 0].to_string()}\n")
    else:
        buf.write(f"Series ({len(val)} éléments)\n{val.head(n_rows).to_string()}")
    return buf.getvalue()


def _dispatch_tool(name: str, args: dict, exec_ns: dict, state) -> tuple[str, list[bytes]]:
    """Redirige un appel d'outil LLM vers la fonction Python correspondante.

    Ce point d'entrée unique simplifie la boucle tool-use dans answer_with_tools() :
    quelle que soit l'évolution future des outils disponibles, la boucle principale
    n'a pas à être modifiée — il suffit d'ajouter un cas ici.
    """
    if name == "execute_python":
        return _exec_python(args.get("code", ""), exec_ns)
    if name == "list_available_data":
        return _list_available_data(exec_ns, state), []
    if name == "get_dataframe_info":
        return _get_dataframe_info(args.get("name", ""), exec_ns, args.get("n_rows", 5)), []
    return f"Outil inconnu : {name}", []


# ─────────────────────────────────────────────────────────────────────────────
# RAG avec tool use
# ─────────────────────────────────────────────────────────────────────────────

def answer_with_tools(
    question: str,
    steps: list[dict],
    exec_ns: dict | None = None,
    state=None,
    top_k: int | None = None,
    summary: str = "",
    system_prompt: str = None,
    conversation_history: list[dict] | None = None,
) -> tuple[str, list[bytes]]:
    """RAG + tool use OpenAI avec fenêtre contextuelle dynamique.

    Le LLM peut appeler execute_python() pour recalculer des indicateurs,
    explorer des DataFrames supprimés, générer des graphiques, etc.
    """
    if exec_ns is None:
        exec_ns = {}

    try:
        client = _get_client()
        sys_prompt = system_prompt or RAG_TOOLS_SYSTEM_PROMPT

        # ── Retrieval RAG avec fenêtre dynamique ──────────────────────────────
        context_parts: list[str] = []
        chunks = build_index(steps)
        if chunks:
            effective_top_k, max_tokens_resp = _dynamic_window(
                chunks, question, summary, sys_prompt
            )
            if top_k is not None:
                effective_top_k = min(top_k, len(chunks))
            context_parts = _retrieve(question, chunks, client, effective_top_k, state)
        else:
            _, max_tokens_resp = _dynamic_window([], question, summary, sys_prompt)

        if summary and summary.strip():
            context_parts.append(f"[Synthèse]\n{summary.strip()}")

        # Résumé du namespace toujours inclus
        ns_summary = _list_available_data(exec_ns, state)
        context_parts.append(f"[Namespace disponible]\n{ns_summary}")

        context = "\n\n---\n\n".join(context_parts) if context_parts else "Aucun contexte."

        # ── Messages initiaux ─────────────────────────────────────────────────
        user_content = (
            f"Contexte de l'analyse :\n\n{context}"
            f"\n\n---\n\nQuestion : {question}"
        )
        messages: list[dict] = [{"role": "system", "content": sys_prompt}]
        for msg in (conversation_history or [])[:-1]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_content})

        # ── Boucle tool use (max 6 tours) ─────────────────────────────────────
        all_figures: list[bytes] = []
        for _ in range(6):
            response = client.chat.completions.create(
                model=config.FORMATTER_MODEL,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                max_tokens=max_tokens_resp,
                temperature=0.2,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                return (msg.content or "").strip(), all_figures

            # Ajouter la réponse assistant (avec tool_calls)
            messages.append(msg)

            # Exécuter chaque outil et renvoyer le résultat
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result_text, figs = _dispatch_tool(tc.function.name, args, exec_ns, state)
                all_figures.extend(figs)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text[:5000],
                })

        return "Limite de tours atteinte — veuillez reformuler.", all_figures

    except Exception as exc:
        return f"Erreur RAG : {exc}", []
