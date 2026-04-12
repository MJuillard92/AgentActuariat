"""
report_agent/writer_agent.py
WriterAgent — orchestrateur maître avec tool-calling OpenAI.

Rôle : comprendre le besoin de l'utilisateur, appeler les tools actuariels
(statistical_analysis, build_pdf) et rédiger l'analyse. Ne crée pas de
nouvelles fonctions actuarielles — il orchestre celles définies dans tools/.

Cycle de vie :
  1. run_agent_loop(history, df, data_store)
       Boucle tool-calling jusqu'à completion.
       Yield des events : tool_call | tool_result | message | done | error
"""
from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path
from typing import Generator, Any

import pandas as pd

from report_agent.tools.tool_registry import get_capabilities, get_openai_tools, call_tool


class WriterAgent:
    """Orchestrateur maître — dialogue + tool-calling + rédaction."""

    MAX_STEPS = 20   # limite de sécurité pour la boucle

    def __init__(self, model: str = "gpt-4o"):
        self._model = model
        self._client = None   # lazy

    # ── Client OpenAI (lazy) ──────────────────────────────────────────────────

    @property
    def _llm(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI()
        return self._client

    # ── Boucle principale ─────────────────────────────────────────────────────

    def run_agent_loop(
        self,
        history: list[dict],
        df: pd.DataFrame | None = None,
        data_store: dict | None = None,
        csv_path: str | None = None,
        context_docs: list[dict] | None = None,
        step_by_step: bool = False,
        approval_event: threading.Event | None = None,
        cancel_flag: list[bool] | None = None,
    ) -> Generator[dict, None, None]:
        """
        Boucle agent avec tool-calling OpenAI.

        Args:
            history        : historique du dialogue [{role, content}]
            df             : DataFrame du portefeuille (pour statistical_analysis)
            data_store     : résultats accumulés des tool calls précédents
            csv_path       : chemin CSV si df non encore chargé
            context_docs   : liste de {"name": str, "content": str} — docs de référence
            step_by_step   : si True, pause avant chaque exécution de tool (attend approval_event)
            approval_event : threading.Event partagé avec l'UI — set() = approuver
            cancel_flag    : liste mutable [bool] — cancel_flag[0] = True = annuler l'étape

        Yields events :
            {"type": "tool_call",        "tool": str, "function_name": str, "params": dict}
            {"type": "awaiting_approval","tool": str, "function_name": str, "params": dict}
            {"type": "tool_result",      "tool": str, "function_name": str, "result": dict}
            {"type": "message",          "content": str}
            {"type": "done"}
            {"type": "error",            "message": str}
        """
        if data_store is None:
            data_store = {}

        # Charge le CSV si df non fourni
        if df is None and csv_path:
            try:
                df = self._load_csv(csv_path)
            except Exception as exc:
                yield {"type": "error", "message": f"Impossible de charger le CSV : {exc}"}
                return

        system_prompt = self._build_system_prompt(df, context_docs=context_docs)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._format_history(history))

        tools = get_openai_tools()
        steps = 0

        while steps < self.MAX_STEPS:
            steps += 1
            try:
                response = self._llm.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None,
                    max_tokens=4000,
                )
            except Exception as exc:
                yield {"type": "error", "message": f"Erreur API OpenAI : {exc}"}
                return

            choice = response.choices[0]
            msg = choice.message

            # Ajouter la réponse de l'assistant à l'historique
            messages.append(msg.model_dump(exclude_none=True))

            # ── Cas 1 : appels de tools ───────────────────────────────────────
            if choice.finish_reason == "tool_calls" and msg.tool_calls:
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        fn_args = {}

                    function_name = fn_args.get("function_name", "")
                    params = fn_args.get("params", {})

                    yield {
                        "type": "tool_call",
                        "tool": fn_name,
                        "function_name": function_name,
                        "params": params,
                        "tool_call_id": tc.id,
                    }

                    # Mode pas à pas : pause avant exécution
                    if step_by_step and approval_event is not None:
                        approval_event.clear()
                        yield {
                            "type": "awaiting_approval",
                            "tool": fn_name,
                            "function_name": function_name,
                            "params": params,
                            "tool_call_id": tc.id,
                        }
                        approval_event.wait(timeout=300)
                        if cancel_flag and cancel_flag[0]:
                            cancel_flag[0] = False
                            rejection = {"erreur": "Étape annulée par l'utilisateur."}
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(rejection, ensure_ascii=False),
                            })
                            yield {
                                "type": "tool_result",
                                "tool": fn_name,
                                "function_name": function_name,
                                "result": rejection,
                                "tool_call_id": tc.id,
                            }
                            continue  # passe au tool_call suivant du même batch

                    # Exécution du tool
                    context_for_tool = None
                    if fn_name == "reasoning":
                        context_for_tool = {
                            "user_message": history[-1].get("content", "") if history else "",
                            "history": history,
                            "csv_columns": list(df.columns) if df is not None else [],
                        }
                    result = call_tool(
                        tool_name=fn_name,
                        function_name=function_name,
                        params=params,
                        df=df,
                        data=data_store,
                        context=context_for_tool,
                    )

                    # Stocker le résultat dans data_store pour les tools suivants
                    _RESULT_KEYS = {
                        "portfolio_summary": "summary",
                        "age_distribution":  "ages",
                        "time_series":       "series",
                        "segmentation":      "segmentation",
                        # builder
                        "exposure":          "exposure_table",
                        "crude_rates":       "qx_table",
                        "smoothing":         "smoothed_table",
                        "diagnostics":       "diagnostics",
                        "validation":        "validation",
                        "benchmarking":      "benchmarking",
                    }
                    if "erreur" not in result:
                        store_key = _RESULT_KEYS.get(function_name, function_name)
                        # Pour builder.exposure, stocker exposure_table directement
                        if fn_name == "builder" and function_name == "exposure":
                            data_store["exposure_table"] = result.get("exposure_table", [])
                        elif fn_name == "builder" and function_name == "crude_rates":
                            data_store["qx_table"] = result.get("qx_table", [])
                        elif fn_name == "builder" and function_name == "smoothing":
                            data_store["smoothed_table"] = result.get("smoothed_table", [])
                        else:
                            data_store[store_key] = result

                    # Enregistrer dans le log de session
                    data_store.setdefault("_call_log", [])
                    data_store["_call_log"].append({
                        "step":          len(data_store["_call_log"]) + 1,
                        "tool":          fn_name,
                        "function_name": function_name,
                        "params":        params,
                        "result_summary": {
                            k: (f"[{len(v)} lignes]" if isinstance(v, list) else str(v)[:300])
                            for k, v in result.items()
                            if k not in ("image_b64", "samples")
                        },
                        "has_error": "erreur" in result,
                    })

                    yield {
                        "type": "tool_result",
                        "tool": fn_name,
                        "function_name": function_name,
                        "result": result,
                        "tool_call_id": tc.id,
                    }

                    # Ajouter le résultat dans les messages (format OpenAI tool result)
                    # Les images base64 sont tronquées pour ne pas saturer le contexte
                    result_for_msg = {
                        k: ("<image base64 tronquée>" if k == "image_b64" else v)
                        for k, v in result.items()
                    }
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result_for_msg, ensure_ascii=False, default=str)[:6000],
                    })

            # ── Cas 2 : réponse texte ─────────────────────────────────────────
            else:
                content = msg.content or ""
                if content:
                    data_store.setdefault("_reasoning_log", [])
                    data_store["_reasoning_log"].append(content)
                yield {"type": "message", "content": content}

                # L'agent signale qu'il a terminé
                if choice.finish_reason in ("stop", "length") or "<FIN>" in content:
                    yield {"type": "done"}
                    return

        # Sécurité : limite d'étapes atteinte
        yield {"type": "error", "message": f"Limite de {self.MAX_STEPS} étapes atteinte."}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_system_prompt(
        self,
        df: "pd.DataFrame | None" = None,
        context_docs: "list[dict] | None" = None,
    ) -> str:
        """
        Charge le prompt système.

        Si loader.py existe à la racine du projet, utilise get_system_prompt()
        pour assembler le prompt depuis system_prompt_level1.md + agent_instructions/*.md
        + catalogue.yaml.

        Sinon, retombe sur writer_dialog_prompt.md (comportement original).
        """
        # ── Try loader.py (new architecture) ──────────────────────────────────
        project_root = Path(__file__).parent.parent
        loader_path = project_root / "loader.py"
        if loader_path.exists():
            try:
                import importlib.util as _ilu
                spec = _ilu.spec_from_file_location("loader", loader_path)
                loader_mod = _ilu.module_from_spec(spec)
                spec.loader.exec_module(loader_mod)
                base = loader_mod.get_system_prompt()
            except Exception:
                # Fallback silently to writer_dialog_prompt.md on any error
                prompt_path = Path(__file__).parent / "writer_dialog_prompt.md"
                base = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
        else:
            # ── Fallback: original writer_dialog_prompt.md ─────────────────────
            prompt_path = Path(__file__).parent / "writer_dialog_prompt.md"
            base = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

        caps = get_capabilities()
        caps_text = json.dumps(caps, ensure_ascii=False, indent=2)
        prompt = (
            base
            + "\n\n## Catalogue de tes tools (builder_capabilities.json)\n\n"
            + "```json\n" + caps_text + "\n```\n"
        )

        if df is not None:
            from report_agent.dictionary.column_schema import build_mapping_report
            report = build_mapping_report(df, caps)

            prompt += f"\n\n## Données du portefeuille chargées — {len(df):,} lignes, {len(df.columns)} colonnes\n\n"

            # ── Mapping automatique ───────────────────────────────────────────
            prompt += "### Mapping automatique des colonnes\n\n"
            prompt += "| Rôle | Colonne détectée | Statut |\n|---|---|---|\n"
            from report_agent.dictionary.column_schema import COLUMN_SCHEMA
            for role, info in COLUMN_SCHEMA.items():
                if role in report["matched"]:
                    col = report["matched"][role]
                    prompt += f"| {info['label']} | `{col}` | ✓ auto |\n"
                else:
                    prompt += f"| {info['label']} | — | ❌ absent |\n"

            # ── Colonnes non reconnues ────────────────────────────────────────
            if report["unknown_cols"]:
                cols_str = ", ".join(f"`{c}`" for c in report["unknown_cols"])
                prompt += f"\n**Colonnes non reconnues** (rôle inconnu — à clarifier) : {cols_str}\n"

            # ── Disponibilité par fonction ────────────────────────────────────
            prompt += "\n### Disponibilité des fonctions\n\n"
            prompt += "| Fonction | Prêt | Colonnes requises manquantes | Notes |\n|---|---|---|---|\n"
            for fn_name, status in report["fn_readiness"].items():
                ready_icon = "✓" if status["ready"] else "⚠"
                missing = ", ".join(status["missing_required"]) if status["missing_required"] else "—"
                missing_opt = ", ".join(status["missing_optional"]) if status["missing_optional"] else ""
                note = f"Optionnel absent : {missing_opt}" if missing_opt else "—"
                prompt += f"| `{fn_name}` | {ready_icon} | {missing} | {note} |\n"

            # ── Questions à poser si colonnes requises manquantes ─────────────
            all_missing_req = {
                role
                for s in report["fn_readiness"].values()
                for role in s["missing_required"]
            }
            if all_missing_req:
                prompt += "\n### Colonnes requises non détectées — questions à poser au client\n\n"
                for role in all_missing_req:
                    info = report["unmatched"].get(role, {})
                    prompt += f"- **{info.get('label', role)}** : {info.get('question', '')}\n"

        # ── Documents de contexte (uploads de l'utilisateur) ─────────────────
        if context_docs:
            prompt += "\n\n## Documents de contexte fournis par l'utilisateur\n\n"
            prompt += (
                "_Ces documents ont été chargés via l'interface. Utilise-les pour enrichir "
                "tes commentaires, comparer avec des résultats antérieurs, ou utiliser une "
                "table de référence personnalisée._\n\n"
            )
            for doc in context_docs:
                prompt += f"### {doc['name']}\n\n```\n{doc['content']}\n```\n\n"

        return prompt

    @staticmethod
    def _format_history(history: list[dict]) -> list[dict]:
        """Convertit l'historique canvas en messages OpenAI."""
        msgs = []
        for h in history:
            role = h.get("role", "user")
            # Normaliser les rôles canvas → OpenAI
            if role in ("assistant_rag", "assistant"):
                role = "assistant"
            elif role != "user":
                role = "user"
            content = h.get("content", "")
            if content:
                msgs.append({"role": role, "content": str(content)})
        return msgs

    @staticmethod
    def _load_csv(csv_path: str) -> pd.DataFrame:
        """Charge un CSV avec détection automatique du séparateur."""
        import csv as _csv
        sep_candidates = [";", ",", "\t", "|"]
        for sep in sep_candidates:
            try:
                df = pd.read_csv(csv_path, sep=sep, encoding="utf-8", engine="python")
                if len(df.columns) > 1:
                    return df
            except Exception:
                pass
        # Fallback latin-1
        for sep in sep_candidates:
            try:
                df = pd.read_csv(csv_path, sep=sep, encoding="latin-1", engine="python")
                if len(df.columns) > 1:
                    return df
            except Exception:
                pass
        raise ValueError(f"Impossible de lire le CSV : {csv_path}")
