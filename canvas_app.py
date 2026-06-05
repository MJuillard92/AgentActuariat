"""
canvas_app.py
Interface principale — 2 onglets :
  • Rapport guidé : dialogue avec le WriterAgent (upload CSV + chat)
  • DEV           : gestion des capacités actuarielles (cards + éditeur de code)
"""
from __future__ import annotations

import base64
import datetime
import io
import json
import threading
from io import StringIO
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import ALL, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from tools.tool_registry import get_capabilities
from agents.mortality.dictionary.column_schema import COLUMN_SCHEMA, build_mapping_report
from knowledge_base.rag_doctrine.manage import ui as doctrine_ui

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="Agent Actuariat",
)
server = app.server

# ─────────────────────────────────────────────────────────────────────────────
# Thread state (WriterAgent)
# ─────────────────────────────────────────────────────────────────────────────

_SESSIONS_DIR = Path(__file__).parent / "session" / "data"

_writer_state: dict = {
    "events": [], "running": False, "data_store": {}, "context_docs": [],
    "step_by_step": False, "pending_tool_call": None,
    "session_id": None,       # yymmddhhmm — set on first tool call
    "csv_filename": None,
}
_writer_lock = threading.Lock()


def _new_session_id() -> str:
    return datetime.datetime.now().strftime("%y%m%d%H%M")


def restore_session(session_id: str) -> tuple[str, list[dict]]:
    """
    Restaure une session depuis le SessionState persisté (MemoryManager).
    Retourne (message_statut, historique_chat_initial).
    """
    from session.memory_manager import MemoryManager
    mm = MemoryManager(session_id)
    mm.load()

    if not mm.state.tool_results and not mm.state.study_plan.is_complete():
        return f"Session introuvable ou vide : {session_id}", []

    data_store = mm.to_data_store()

    with _writer_lock:
        _writer_state["data_store"]   = data_store
        _writer_state["session_id"]   = session_id
        _writer_state["csv_filename"] = mm.state.csv_filename

    # Message de contexte pour l'agent
    key_labels = {
        "exposure_table":       "Table d'exposition",
        "qx_table":             "Taux bruts",
        "smoothed_table":       "Table lissée",
        "diagnostics":          "Diagnostics de crédibilité",
        "validation":           "Validation statistique",
        "benchmarking":         "Benchmarking",
        "certification_report": "Rapport PDF",
    }
    computed = [label for key, label in key_labels.items()
                if mm.state.tool_results.get(key)]
    n_calls = len(data_store.get("_call_log", []))

    lines = [
        f"[Session restaurée : {session_id}]",
        f"Calculs disponibles : {', '.join(computed) or 'aucun'}",
        f"Appels tools : {n_calls}",
    ]
    if mm.state.context_summary:
        lines.append(mm.state.context_summary.to_system_block())
    lines.append(
        "\nVous pouvez poser des questions sur ces résultats ou "
        "demander la génération du rapport de certification."
    )

    status = (
        f"Session {session_id} restaurée — "
        f"{len(computed)} calculs, {n_calls} appels"
    )
    return status, [{"role": "assistant", "content": "\n".join(lines)}]


def list_sessions() -> list[dict]:
    """Retourne la liste des sessions disponibles, triée par date décroissante."""
    from session.session_state import SessionState
    if not _SESSIONS_DIR.exists():
        return []
    sessions = []
    for p in sorted(_SESSIONS_DIR.glob("*_state.json"), reverse=True):
        try:
            state = SessionState.model_validate_json(p.read_text(encoding="utf-8"))
            sessions.append({
                "session_id":   state.session_id,
                "timestamp":    state.updated_at[:16].replace("T", " "),
                "csv_filename": state.csv_filename or "—",
                "n_tool_calls": len(state.tool_results),
            })
        except Exception:
            continue
    return sessions

# Synchronisation mode pas à pas
_step_approval_event: threading.Event = threading.Event()
_step_cancel_flag: list[bool] = [False]


def _run_writer_in_thread(history: list[dict], df_json: str | None) -> None:
    from agents.mortality.agents.graph import stream_agent

    # Récupérer le data_store et context_docs persistés de la session
    with _writer_lock:
        data_store   = _writer_state["data_store"]
        context_docs = _writer_state["context_docs"]
        step_by_step = _writer_state["step_by_step"]
        # Générer un session_id si ce n'est pas encore fait
        if not _writer_state["session_id"]:
            _writer_state["session_id"] = _new_session_id()
        session_id    = _writer_state["session_id"]
        csv_filename  = _writer_state["csv_filename"]

    # Le DataFrame est chargé par MemoryManager depuis Parquet si besoin.
    # On passe df=None sauf si c'est le premier tour après upload (dataset pas encore enregistré).
    df = None
    if df_json:
        from session.dataset_store import DatasetStore
        if not DatasetStore.exists(session_id):
            # Premier tour après upload — enregistrement via stream_agent → mm.register_dataset
            try:
                df = pd.read_json(StringIO(df_json), orient="split")
            except Exception:
                pass

    if step_by_step:
        _step_approval_event.clear()
        _step_cancel_flag[0] = False

    # Séparateur de tour dans le log internals
    last_msg = history[-1].get("content", "")[:80] if history else ""
    with _writer_lock:
        _writer_state["events"].append({
            "type": "new_turn",
            "user_msg": last_msg,
        })

    # Activer le hub MasterAgent seulement si pas d'agent déjà défini
    # (submit_disambiguation peut avoir injecté "builder" directement)
    if "_initial_active_agent" not in data_store:
        data_store["_initial_active_agent"] = "master"

    # ── Audit JSON (piste humaine — jamais lu par l'agent) ───────────────────
    _audit_path = _SESSIONS_DIR / f"{session_id}_audit.json"
    _audit_entries: list[dict] = []

    def _append_audit(ev: dict) -> None:
        _audit_entries.append({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            **{k: v for k, v in ev.items() if k not in ("image_b64",)},
        })
        try:
            _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            _audit_path.write_text(
                json.dumps(_audit_entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    try:
        for event in stream_agent(
            history, df=df, data_store=data_store, context_docs=context_docs,
            step_by_step=step_by_step,
            approval_event=_step_approval_event if step_by_step else None,
            cancel_flag=_step_cancel_flag if step_by_step else None,
            thread_id=session_id,
        ):
            _append_audit(event)
            with _writer_lock:
                _writer_state["events"].append(event)
                if event["type"] == "awaiting_approval":
                    _writer_state["pending_tool_call"] = {
                        "tool": event.get("tool"),
                        "function_name": event.get("function_name"),
                        "params": event.get("params", {}),
                    }
                elif event["type"] in ("tool_result", "done", "error"):
                    _writer_state["pending_tool_call"] = None
                # La persistance est gérée par MemoryManager.after_turn() dans stream_agent()
    except Exception as exc:
        _append_audit({"type": "error", "message": str(exc)})
        with _writer_lock:
            _writer_state["events"].append({"type": "error", "message": str(exc)})
    finally:
        with _writer_lock:
            _writer_state["running"] = False
            _writer_state["pending_tool_call"] = None
            # Effacer le bypass MasterAgent pour que le tour suivant repasse par master
            _writer_state["data_store"].pop("_initial_active_agent", None)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pending_banner(pending: dict) -> html.Div:
    """Bannière affichée quand l'agent attend une approbation (mode pas à pas)."""
    tool = pending.get("tool", "")
    fn   = pending.get("function_name", "")
    params = pending.get("params", {})
    return dbc.Alert([
        html.Div([
            html.I(className="fa fa-pause-circle me-2"),
            html.Strong(f"Prochaine action : {tool}.{fn}"),
            dbc.Badge("en attente", color="warning", className="ms-2"),
        ], className="mb-2"),
        html.Pre(
            json.dumps(params, ensure_ascii=False, indent=2),
            className="small mb-2",
            style={"background": "#fff8e1", "padding": "8px", "borderRadius": "4px",
                   "maxHeight": "120px", "overflowY": "auto", "fontSize": "11px"},
        ),
        dbc.Row([
            dbc.Col(dbc.Button(
                [html.I(className="fa fa-play me-1"), "Exécuter"],
                id="btn-step-approve", color="success", size="sm", n_clicks=0,
            ), width="auto"),
            dbc.Col(dbc.Button(
                [html.I(className="fa fa-times me-1"), "Annuler cette étape"],
                id="btn-step-cancel", color="outline-danger", size="sm", n_clicks=0,
            ), width="auto"),
        ], className="g-2"),
    ], color="warning", className="mb-0 rounded-0 border-start-0 border-end-0")


def _parse_csv(contents: str, filename: str) -> tuple:
    """Décode le contenu base64 d'un dcc.Upload et retourne (df, erreur)."""
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        for sep in (";", ",", "\t", "|"):
            for enc in ("utf-8", "latin-1"):
                try:
                    df = pd.read_csv(io.BytesIO(decoded), sep=sep, encoding=enc, engine="python")
                    if len(df.columns) > 1:
                        return df, ""
                except Exception:
                    pass
        return None, f"Impossible de lire {filename}"
    except Exception as exc:
        return None, str(exc)


def _mapping_badge(df: pd.DataFrame) -> dbc.ListGroup:
    """Résumé du mapping colonnes."""
    caps = get_capabilities()
    report = build_mapping_report(df, caps)
    items = []
    for role, info in COLUMN_SCHEMA.items():
        if role in report["matched"]:
            items.append(dbc.ListGroupItem(
                [html.I(className="fa fa-check-circle text-success me-2"),
                 html.Span(info["label"], className="fw-bold"),
                 html.Span(f" → {report['matched'][role]}", className="text-muted small")],
                className="py-1 px-2",
            ))
        else:
            items.append(dbc.ListGroupItem(
                [html.I(className="fa fa-times-circle text-danger me-2"),
                 html.Span(info["label"], className="text-muted")],
                className="py-1 px-2",
            ))
    if report["unknown_cols"]:
        items.append(dbc.ListGroupItem(
            [html.I(className="fa fa-question-circle text-warning me-2"),
             html.Span(f"Colonnes non reconnues : {', '.join(report['unknown_cols'])}", className="small text-muted")],
            className="py-1 px-2",
        ))
    return dbc.ListGroup(items, flush=True, className="small")


def _format_clone_message(audit: dict) -> str:
    """Message visible décrivant la base de travail créée + les modifications.
    `audit` = data_store["_audit"]["normalization"] produit par
    maybe_normalize_records."""
    lines = [
        "✅ Base de travail créée (clone normalisé — le fichier original "
        "reste intact).",
        "",
        "Modifications appliquées :",
    ]
    col_map = audit.get("column_mapping") or {}
    renames = [f"{csv} → {canon}" for canon, csv in col_map.items() if csv]
    if renames:
        lines.append("• Colonnes renommées : " + ", ".join(renames))
    val_map = audit.get("value_mapping") or {}
    vparts = []
    for col, mp in val_map.items():
        if isinstance(mp, dict) and mp:
            vparts.append(f"{col} (" + ", ".join(f"{o}→{c}" for o, c in mp.items()) + ")")
    if vparts:
        lines.append("• Valeurs normalisées : " + " ; ".join(vparts))
    lines.append(
        "• Dates parsées (format JJ/MM/AAAA) ; sentinelles (31/12/2999, …) "
        "traitées comme contrats actifs"
    )
    ri, ro = audit.get("rows_in"), audit.get("rows_out")
    if ri is not None and ro is not None:
        txt = (f"• {ri:,} lignes en entrée → {ro:,} lignes dans la base de "
               f"travail").replace(",", " ")
        lines.append(txt)
    lines.append("")
    lines.append("Vous pouvez maintenant lancer vos calculs.")
    return "\n".join(lines)


def _build_methods_selects(form_id: str) -> list:
    """Construit les selects par tool à inclure dans la bulle data catalogue.

    Lit le catalogue des tools via `all_choices_for_mode("full_report")` —
    couvre les 3 tools usuels (crude_rates, smoothing, validation). Les
    valeurs ne sont prises en compte au submit que si methods-mode=explicit.
    """
    try:
        from agents.master.method_choices import all_choices_for_mode
        choices = all_choices_for_mode("full_report") or []
    except Exception:
        choices = []
    if not choices:
        return [html.Div("Aucune méthode configurable.",
                         className="text-muted small fst-italic")]
    rendered = []
    for c in choices:
        rendered.append(html.Div([
            dbc.Label(f"{c.label} :", className="small fw-bold mt-1"),
            dbc.Select(
                id={"type": "dcf-method", "form_id": form_id, "tool": c.tool},
                options=[{"label": v, "value": v} for v in c.choices],
                value=c.default,
                size="sm",
            ),
        ], className="mb-1"))
    return rendered


def _render_datacatalogue_bubble(entry: dict) -> html.Div:
    """Bulle inline qui contient le formulaire « Compléter le data catalogue ».

    Affichée comme message de l'assistant quand le Builder gate refuse
    (event datacatalogue_incomplete). L'utilisateur remplit puis confirme
    sans quitter la conversation — alternative MCP-UI au modal popup.

    `entry` doit contenir : form_id, missing, suggestions, submitted.
    Si submitted=True, la bulle se replie en mode confirmation figée.
    """
    form_id = entry.get("form_id", "dc-form-unknown")
    submitted = entry.get("submitted", False)
    summary = entry.get("submitted_summary", "")

    if submitted:
        bubble = dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fa fa-check-circle me-2 text-success"),
                    html.Strong("Data catalogue complété "),
                    html.Span(f"— {summary}", className="text-muted small")
                    if summary else None,
                ]),
            ], className="py-2"),
        ], color="success", outline=True, className="mb-0")
        return html.Div(
            bubble,
            className="d-flex mb-3 justify-content-start",
            style={"maxWidth": "80%"},
        )

    sug = entry.get("suggestions") or {}
    auto_start = sug.get("start_year")
    auto_end = sug.get("end_year")

    def _id(field: str) -> dict:
        return {"type": "dcf", "f": field, "form_id": form_id}

    bubble = dbc.Card([
        dbc.CardHeader([
            html.I(className="fa fa-clipboard-list me-2 text-warning"),
            html.Strong("Compléter le data catalogue"),
        ], className="py-2"),
        dbc.CardBody([
            html.P(
                "Avant de lancer les calculs j'ai besoin de quelques précisions. "
                "Renseignez tous les champs ci-dessous puis cliquez sur Confirmer.",
                className="text-muted small mb-3",
            ),
            # ── Périmètre ─────────────────────────────────────────────
            html.H6([html.I(className="fa fa-bullseye me-2 text-primary"),
                     "Périmètre"], className="text-secondary mt-2 small fw-bold"),
            dbc.Label("Mode de rapport :", className="small fw-bold mt-2"),
            dbc.RadioItems(
                id=_id("report-mode"),
                options=[
                    {"label": " Rapport complet (taux bruts + lissage + validation + benchmarking)",
                     "value": "full_report"},
                    {"label": " Taux bruts uniquement (sans lissage)",
                     "value": "raw_rates"},
                    {"label": " Description du portefeuille seule",
                     "value": "description"},
                ],
                value="full_report", inline=False,
            ),
            dbc.Label("Segmentation par sexe :", className="small fw-bold mt-2"),
            dbc.RadioItems(
                id=_id("gender"),
                options=[
                    {"label": " Table unisex (agrégée)",          "value": "unisex"},
                    {"label": " Tables séparées Hommes / Femmes", "value": "by_sex"},
                ],
                value="unisex", inline=True,
            ),
            dbc.Label("Générer un rapport PDF en fin de calcul ?",
                      className="small fw-bold mt-2"),
            dbc.RadioItems(
                id=_id("write"),
                options=[
                    {"label": " Oui, PDF + notebook", "value": "yes"},
                    {"label": " Non, juste les calculs", "value": "no"},
                ],
                value="yes", inline=True,
            ),
            # ── Méthodes ──────────────────────────────────────────────
            html.H6([html.I(className="fa fa-cogs me-2 text-primary"),
                     "Méthodes"], className="text-secondary mt-3 small fw-bold"),
            dbc.RadioItems(
                id=_id("methods-mode"),
                options=[
                    {"label": " Mode auto (le système choisit)",      "value": "auto"},
                    {"label": " Préciser chaque méthode manuellement", "value": "explicit"},
                ],
                value="auto", inline=False,
            ),
            # Selects par tool — toujours affichés mais ignorés si mode=auto.
            # Liste dérivée DYNAMIQUEMENT du catalogue des tools via
            # all_choices_for_mode (full_report unisex par défaut — l'user
            # peut adapter ensuite, on relit son choix au submit).
            html.Div(_build_methods_selects(form_id),
                     className="ms-3 mt-2",
                     style={"borderLeft": "2px solid #FFD580",
                            "paddingLeft": "10px"}),
            # ── Période d'observation ─────────────────────────────────
            html.H6([html.I(className="fa fa-calendar me-2 text-primary"),
                     "Période d'observation"],
                    className="text-secondary mt-3 small fw-bold"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Année de début :", className="small fw-bold"),
                    dbc.Input(id=_id("start-year"), type="number",
                              value=auto_start,
                              placeholder="ex : 1983", size="sm"),
                ], md=6),
                dbc.Col([
                    dbc.Label("Année de fin :", className="small fw-bold"),
                    dbc.Input(id=_id("end-year"), type="number",
                              value=auto_end,
                              placeholder="ex : 2010", size="sm"),
                ], md=6),
            ], className="mt-1"),
            # ── Bouton confirmer ──────────────────────────────────────
            html.Div([
                dbc.Button(
                    [html.I(className="fa fa-check me-1"), "Confirmer"],
                    id={"type": "dcf-confirm", "form_id": form_id},
                    color="primary", size="sm", n_clicks=0,
                    className="mt-3",
                ),
            ], className="d-flex justify-content-end"),
        ], className="py-3"),
    ], color="warning", outline=True, className="mb-0")

    return html.Div(
        bubble,
        className="d-flex mb-3 justify-content-start",
        style={"maxWidth": "80%"},
    )


def _render_decision_required_bubble(entry: dict) -> html.Div:
    """Bulle inline « décision requise » pour les tools qui retournent
    `decision_required` (ex. smoothing → violations de monotonie).

    `entry` doit contenir : form_id, tool, reason, options[], submitted.
    Si submitted=True, la bulle se replie avec le choix retenu.

    Plan refonte garde-fou 2026-06-03.
    """
    form_id = entry.get("form_id", "dr-form-unknown")
    submitted = entry.get("submitted", False)
    chosen = entry.get("chosen", "")
    tool = entry.get("tool", "")
    reason = entry.get("reason", "")
    options = entry.get("options") or []

    if submitted:
        chosen_label = next(
            (o.get("label", chosen) for o in options if o.get("id") == chosen),
            chosen,
        )
        bubble = dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fa fa-check-circle me-2 text-success"),
                    html.Strong("Décision enregistrée "),
                    html.Span(f"— {chosen_label}", className="text-muted small"),
                ]),
            ], className="py-2"),
        ], color="success", outline=True, className="mb-0")
        return html.Div(
            bubble,
            className="d-flex mb-3 justify-content-start",
            style={"maxWidth": "80%"},
        )

    radio_options = [
        {"label": f" {o.get('label', o.get('id', '?'))}",
         "value": o.get("id", "")}
        for o in options if o.get("id")
    ]

    bubble = dbc.Card([
        dbc.CardHeader([
            html.I(className="fa fa-pause-circle me-2 text-warning"),
            html.Strong(f"Décision requise — {tool or 'outil de calcul'}"),
        ], className="py-2"),
        dbc.CardBody([
            html.P(reason, className="small mb-3") if reason else None,
            dbc.RadioItems(
                id={"type": "dr-option", "form_id": form_id},
                options=radio_options,
                value=(radio_options[0]["value"] if radio_options else ""),
                inline=False,
                className="small",
            ),
            html.Div([
                dbc.Button(
                    [html.I(className="fa fa-check me-1"), "Confirmer"],
                    id={"type": "dr-submit", "form_id": form_id},
                    color="primary", size="sm", n_clicks=0,
                    className="mt-2",
                ),
            ], className="d-flex justify-content-end"),
        ], className="py-3"),
    ], color="warning", outline=True, className="mb-0")

    return html.Div(
        bubble,
        className="d-flex mb-3 justify-content-start",
        style={"maxWidth": "80%"},
    )


def _chat_bubble(role: str, content: str, extra: dict | None = None) -> html.Div:
    """Rend une bulle de chat."""
    is_user = role == "user"
    extra_children = []

    if extra and extra.get("type") == "tool_call":
        fn = extra.get("function_name", "")
        tool = extra.get("tool", "")
        extra_children = [
            html.Div(
                [html.I(className="fa fa-cog fa-spin me-1 text-warning"),
                 html.Span(f"{tool}.{fn}", className="fw-bold small text-warning")],
                className="mb-1",
            )
        ]
    elif extra and extra.get("type") == "tool_result" and extra.get("table"):
        # Tableau de données tabulaires
        rows = extra["table"]
        headers = extra.get("columns_header") or (list(rows[0].keys()) if rows else [])
        thead = html.Thead(html.Tr([html.Th(h, className="small") for h in headers]))
        tbody_rows = []
        for row in rows[:20]:
            cells = [html.Td(str(row.get(h, "")), className="small") for h in headers]
            tbody_rows.append(html.Tr(cells))
        extra_children = [
            dbc.Table(
                [thead, html.Tbody(tbody_rows)],
                bordered=True, size="sm", hover=True, responsive=True,
                className="mt-2 mb-0",
                style={"fontSize": "11px"},
            )
        ]
    elif extra and extra.get("type") == "tool_result" and extra.get("samples"):
        # Galerie multi-images
        valid = [s for s in extra["samples"] if s.get("image_b64")]
        cols = []
        for s in valid:
            cols.append(dbc.Col([
                html.P(s.get("title", ""), className="small fw-bold mb-0 text-center"),
                html.Img(
                    src=f"data:image/png;base64,{s['image_b64']}",
                    style={"width": "100%", "borderRadius": "4px"},
                ),
                html.P(s.get("description", ""), className="small text-muted text-center mb-1"),
            ], width=6, className="mb-2"))
        extra_children = [dbc.Row(cols)]
    elif extra and extra.get("type") == "tool_result" and extra.get("image_b64"):
        extra_children = [
            html.Img(
                src=f"data:image/png;base64,{extra['image_b64']}",
                style={"maxWidth": "100%", "borderRadius": "6px", "marginTop": "8px"},
            )
        ]
    elif extra and extra.get("type") == "tool_result":
        fn = extra.get("function_name", "")
        keys = extra.get("result_keys", [])
        extra_children = [
            html.Div(
                [html.I(className="fa fa-check-circle me-1 text-success"),
                 html.Span(f"{fn} → {', '.join(keys)}", className="small text-muted")],
                className="mb-1",
            )
        ]

    # Bulle spéciale : désambiguation en attente
    if role == "_disambiguation":
        return html.Div(
            dbc.Alert([
                html.I(className="fa fa-clipboard-check me-2 text-primary"),
                html.Strong("Informations requises"),
                html.Span(" — remplissez le formulaire ci-dessous pour lancer l'analyse.",
                          className="ms-1 text-muted"),
                dbc.Button(
                    [html.I(className="fa fa-edit me-1"), "Ouvrir le formulaire"],
                    id="btn-open-disambiguation",
                    color="primary",
                    size="sm",
                    n_clicks=0,
                    className="ms-3",
                ),
            ], color="info", className="mb-0 py-2"),
            className="d-flex mb-3 justify-content-start",
        )

    bubble = html.Div(
        extra_children + ([dcc.Markdown(content, className="mb-0")] if content else []),
        className="p-3 rounded",
        style={
            "background": "#DCF8C6" if is_user else "#FFFFFF",
            "border": "1px solid #E0E0E0",
            "maxWidth": "80%",
        },
    )
    return html.Div(
        bubble,
        className="d-flex mb-3 " + ("justify-content-end" if is_user else "justify-content-start"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab : Rapport guidé
# ─────────────────────────────────────────────────────────────────────────────

def _writer_tab() -> html.Div:
    return html.Div([
        dbc.Row([
            # ── Panneau gauche : Outils ──────────────────────────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fa fa-file-csv me-2"),
                        html.Strong("Portefeuille CSV"),
                    ]),
                    dbc.CardBody([
                        dcc.Upload(
                            id="upload-csv",
                            children=html.Div([
                                html.I(className="fa fa-upload me-2 text-muted"),
                                html.Span("Glisser-déposer ou "),
                                html.A("choisir un fichier"),
                            ]),
                            style={
                                "borderWidth": "2px",
                                "borderStyle": "dashed",
                                "borderRadius": "8px",
                                "borderColor": "#CCCCCC",
                                "textAlign": "center",
                                "padding": "14px",
                                "cursor": "pointer",
                                "backgroundColor": "#FAFAFA",
                            },
                            multiple=False,
                        ),
                        html.Div(id="csv-info", className="mt-2"),
                    ]),
                ], className="mb-2"),

                # ── Rapports générés (dossier session/rapports/) ───────────
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fa fa-folder-open me-2"),
                        html.Strong("Rapports"),
                        dbc.Button(
                            html.I(className="fa fa-sync"),
                            id="btn-rapports-refresh",
                            color="link", size="sm",
                            className="float-end p-0",
                            title="Rafraîchir la liste",
                        ),
                    ]),
                    dbc.CardBody([
                        html.Div(id="rapports-list",
                                 style={"maxHeight": "260px",
                                        "overflowY": "auto"}),
                    ]),
                ], className="mb-2"),

                # ── Reprendre une session ─────────────────────────────────
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fa fa-history me-2"),
                        html.Strong("Session"),
                    ]),
                    dbc.CardBody([
                        dbc.InputGroup([
                            dbc.Input(
                                id="input-session-id",
                                placeholder="ex : 2604021636",
                                size="sm",
                                debounce=False,
                            ),
                            dbc.Button(
                                [html.I(className="fa fa-redo me-1"), "Reprendre"],
                                id="btn-restore-session",
                                color="outline-secondary",
                                size="sm",
                                n_clicks=0,
                            ),
                        ], className="mb-1"),
                        html.Div(id="restore-session-info", className="small"),
                    ], className="py-2"),
                ], className="mb-2"),

                # ── État data catalogue (3 niveaux) ─────────────────────
                # Plan refonte garde-fou 2026-06-03 (Partie A).
                # ⚪ pas requis (par défaut, l'agent n'a rien demandé)
                # 🟡 à compléter (bulle inline ouverte)
                # 🟢 prêt (compute_datacatalogue_state.complete == True)
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(id="dc-icon",
                                   className="fa fa-circle-question text-muted me-2"),
                            html.Strong("Data catalogue",
                                        className="small"),
                            html.Span(id="dc-status-label",
                                      children=" · pas requis",
                                      className="small text-muted ms-1"),
                        ], id="dc-status-row",
                           style={"cursor": "pointer"},
                           title="État du data catalogue (cliquez pour "
                                 "ouvrir le formulaire)"),
                    ], className="py-2"),
                ], id="card-dc-status", className="mb-2"),

                # ── Mode pas à pas ───────────────────────────────────────
                dbc.Card([
                    dbc.CardBody([
                        dbc.Switch(
                            id="switch-step-mode",
                            label="Mode pas à pas",
                            value=False,
                            className="mb-0",
                        ),
                    ], className="py-2"),
                ], className="mb-2"),

                # ── Documents de contexte ─────────────────────────────────
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fa fa-folder-open me-2"),
                        html.Strong("Documents de contexte"),
                    ]),
                    dbc.CardBody([
                        dcc.Upload(
                            id="upload-context",
                            children=html.Div([
                                html.I(className="fa fa-file-alt me-2 text-muted"),
                                html.Span("PDF / CSV / TXT"),
                            ]),
                            style={
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "6px",
                                "borderColor": "#BBBBBB",
                                "textAlign": "center",
                                "padding": "8px",
                                "cursor": "pointer",
                                "backgroundColor": "#FAFAFA",
                                "fontSize": "12px",
                            },
                            multiple=True,
                        ),
                        html.Div(id="context-docs-list", className="mt-2"),
                    ]),
                ], className="mb-2"),
            ], width=3),

            # ── Panneau central : Internals agent ────────────────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fa fa-brain me-2"),
                        html.Strong("Internals agent"),
                        dbc.Badge("—", id="internals-agent-badge",
                                  color="secondary", className="ms-2"),
                    ]),
                    dbc.CardBody(
                        html.Div(
                            id="agent-internals-log",
                            style={
                                "height": "70vh",
                                "overflowY": "auto",
                                "fontFamily": "monospace",
                                "fontSize": "11px",
                                "background": "#1E1E1E",
                                "color": "#D4D4D4",
                                "padding": "8px",
                                "borderRadius": "4px",
                                "whiteSpace": "pre-wrap",
                                "wordBreak": "break-all",
                            },
                        ),
                        className="p-2",
                    ),
                ]),
            ], width=3),

            # ── Panneau droit : Chat ─────────────────────────────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fa fa-comments me-2"),
                        html.Strong("Dialogue avec l'agent"),
                        dbc.Badge("Prêt", id="agent-status-badge",
                                  color="success", className="ms-2"),
                    ]),
                    dbc.CardBody([
                        html.Div(
                            id="chat-messages",
                            style={"height": "55vh", "overflowY": "auto",
                                   "padding": "8px", "background": "#F5F5F5",
                                   "borderRadius": "6px"},
                        ),
                    ]),
                    html.Div(id="step-approval-banner"),
                    dbc.CardFooter([
                        # Bouton + document mid-conversation
                        html.Div([
                            dcc.Upload(
                                id="upload-mid-chat",
                                children=dbc.Button(
                                    [html.I(className="fa fa-plus me-1"), "Document"],
                                    color="outline-secondary",
                                    size="sm",
                                    style={"fontSize": "12px"},
                                ),
                                multiple=False,
                                accept=".pdf,.csv,.txt,.md",
                                style={"display": "inline-block"},
                            ),
                            html.Span(id="mid-chat-doc-name",
                                      className="ms-2 small text-muted"),
                        ], className="mb-2"),
                        # Zone de saisie
                        dbc.InputGroup([
                            dbc.Textarea(
                                id="chat-input",
                                placeholder="Tapez votre message… (Shift+Entrée pour nouvelle ligne)",
                                style={"resize": "none", "height": "70px"},
                            ),
                            dbc.Button(
                                [html.I(className="fa fa-paper-plane me-1"), "Envoyer"],
                                id="btn-send",
                                color="primary",
                                n_clicks=0,
                            ),
                        ]),
                    ]),
                ]),
            ], width=6),
        ], className="g-3"),
    ], className="p-3")


# ─────────────────────────────────────────────────────────────────────────────
# Tab : DEV
# ─────────────────────────────────────────────────────────────────────────────

def _build_capability_cards() -> list:
    """Construit les cards de capacités depuis le catalogue dynamique (catalogue.py)."""
    caps = get_capabilities()
    cards = []
    for tool_name, tool_info in caps.get("tools", {}).items():
        fn_items = []
        for fn_name, fn_info in tool_info.get("functions", {}).items():
            available = fn_info.get("disponible", True) is not False
            status_badge = (
                dbc.Badge("✓", color="success", className="me-1")
                if available
                else dbc.Badge("indisponible", color="secondary", className="me-1")
            )
            params_text = ""
            if fn_info.get("params"):
                params_text = " | ".join(
                    f"{k}: {v}" for k, v in fn_info["params"].items()
                )
            req_cols = fn_info.get("required_columns", [])
            opt_cols = fn_info.get("optional_columns", [])

            fn_items.append(html.Div([
                html.Div([
                    status_badge,
                    html.Strong(fn_name, className="me-2"),
                    dbc.Button(
                        [html.I(className="fa fa-code me-1"), "Code"],
                        id={"type": "dev-view-code-btn", "tool": tool_name, "fn": fn_name},
                        size="sm", color="outline-secondary", className="me-1",
                        n_clicks=0,
                    ),
                ], className="d-flex align-items-center mb-1"),
                html.P(fn_info.get("description", ""), className="small text-muted mb-1"),
                html.Div([
                    html.Span(f"Req: {', '.join(req_cols)}", className="badge bg-danger me-1") if req_cols else None,
                    html.Span(f"Opt: {', '.join(opt_cols)}", className="badge bg-info me-1") if opt_cols else None,
                    html.Span(params_text, className="small text-secondary") if params_text else None,
                ], className="mb-2"),
                html.Hr(className="my-2"),
            ], className="mb-1"))

        cards.append(dbc.Card([
            dbc.CardHeader([
                html.Strong(tool_name, className="me-2"),
                html.Span(tool_info.get("description", ""), className="small text-muted"),
                dbc.Button(
                    [html.I(className="fa fa-plus me-1"), "Ajouter"],
                    id={"type": "dev-add-fn-btn", "tool": tool_name},
                    size="sm", color="outline-primary", className="ms-auto float-end",
                    n_clicks=0,
                ),
            ]),
            dbc.CardBody(fn_items),
        ], className="mb-3"))
    return cards


def _build_file_tree() -> list:
    """Construit l'arbre de fichiers pour le panneau code."""
    tools_root = Path(__file__).parent / "tools"
    dict_root = Path(__file__).parent / "agents" / "mortality" / "dictionary"
    items = []

    # dictionary/
    items.append(html.Li([
        html.I(className="fa fa-folder-open text-warning me-1"),
        html.Strong("dictionary/"),
    ], className="mt-1"))
    for f in sorted(dict_root.glob("*.py")):
        if f.name.startswith("_"):
            continue
        items.append(html.Li(
            dbc.Button(f.name, id={"type": "dev-file-btn", "path": str(f)},
                       color="link", size="sm", className="py-0 ps-4"),
            className="ms-3",
        ))

    # tools/
    for tool_dir in sorted(tools_root.iterdir()):
        if not tool_dir.is_dir() or tool_dir.name.startswith("_"):
            continue
        items.append(html.Li([
            html.I(className="fa fa-folder-open text-warning me-1"),
            html.Strong(f"tools/{tool_dir.name}/"),
        ], className="mt-2"))
        for f in sorted(tool_dir.glob("*.py")):
            if f.name.startswith("_"):
                continue
            items.append(html.Li(
                dbc.Button(f.name, id={"type": "dev-file-btn", "path": str(f)},
                           color="link", size="sm", className="py-0 ps-4"),
                className="ms-3",
            ))

    return [html.Ul(items, className="list-unstyled small")]


def _new_fn_modal() -> dbc.Modal:
    """Modal pour ajouter une nouvelle fonction à un tool."""
    col_options = [{"label": f"{role} — {info['label']}", "value": role}
                   for role, info in COLUMN_SCHEMA.items()]
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Ajouter une fonction")),
        dbc.ModalBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Tool cible"),
                    dbc.Input(id="new-fn-tool", disabled=True, className="mb-2"),
                    dbc.Label("Nom de la fonction"),
                    dbc.Input(id="new-fn-name", placeholder="ex: compute_lapses", className="mb-2"),
                    dbc.Label("Description"),
                    dbc.Textarea(id="new-fn-desc", rows=2, className="mb-2"),
                    dbc.Label("Colonnes requises"),
                    dcc.Dropdown(id="new-fn-req-cols", options=col_options,
                                 multi=True, className="mb-2"),
                    dbc.Label("Colonnes optionnelles"),
                    dcc.Dropdown(id="new-fn-opt-cols", options=col_options,
                                 multi=True, className="mb-2"),
                    dbc.Label("Paramètres (JSON)"),
                    dbc.Textarea(id="new-fn-params", rows=2, placeholder='{"age_min": "int"}',
                                 className="mb-2"),
                ], width=5),
                dbc.Col([
                    dbc.Label("Code généré (modifiable)"),
                    dbc.Textarea(id="new-fn-code", rows=22,
                                 style={"fontFamily": "monospace", "fontSize": "12px"}),
                ], width=7),
            ]),
            html.Div(id="new-fn-feedback", className="mt-2"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Annuler", id="btn-new-fn-cancel", color="secondary", className="me-2"),
            dbc.Button(
                [html.I(className="fa fa-save me-1"), "Créer la fonction"],
                id="btn-new-fn-create", color="primary",
            ),
        ]),
    ], id="modal-new-fn", size="xl", is_open=False)


def _dev_tab() -> html.Div:
    return html.Div([
        _new_fn_modal(),
        dbc.Tabs([
            dbc.Tab(label="Capacités", tab_id="dev-caps", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="fa fa-sync me-1"), "Rafraîchir"],
                            id="btn-refresh-caps", color="outline-secondary",
                            size="sm", className="mb-3",
                        ),
                        html.Div(id="dev-caps-panel",
                                 children=_build_capability_cards()),
                    ]),
                ]),
            ]),
            dbc.Tab(label="Code", tab_id="dev-code", children=[
                dbc.Row([
                    dbc.Col([
                        html.Div(
                            _build_file_tree(),
                            style={"height": "80vh", "overflowY": "auto",
                                   "borderRight": "1px solid #DDD", "paddingRight": "8px"},
                        ),
                    ], width=3),
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.Input(id="dev-file-path-display", disabled=True,
                                      placeholder="Aucun fichier sélectionné"),
                            dbc.Button(
                                [html.I(className="fa fa-save me-1"), "Sauvegarder"],
                                id="btn-save-code", color="success", size="sm",
                                n_clicks=0,
                            ),
                        ], className="mb-2"),
                        dbc.Textarea(
                            id="dev-code-editor",
                            style={"height": "75vh", "fontFamily": "monospace",
                                   "fontSize": "12px", "resize": "none"},
                            placeholder="Sélectionnez un fichier dans l'arborescence…",
                        ),
                        html.Div(id="dev-save-feedback", className="mt-1 small text-muted"),
                    ], width=9),
                ], className="g-2"),
            ]),
        ], id="dev-tabs", active_tab="dev-caps"),
    ], className="p-3")


# ─────────────────────────────────────────────────────────────────────────────
# Modal désambiguation
# ─────────────────────────────────────────────────────────────────────────────

def _datacatalogue_modal() -> dbc.Modal:
    """Modal « Compléter le data catalogue ».

    Reçoit les choix utilisateur AVANT que le Builder ne tourne — gate
    stricte. Contient les champs requis par `compute_datacatalogue_state` :
      - Périmètre : report_mode, gender_segmentation, write (PDF ?)
      - Méthodes : auto OU explicites (1 select par tool)
      - Période d'observation : start/end year, période

    Plan datacatalogue-gate 2026-05-25.
    """
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            html.I(className="fa fa-clipboard-list me-2 text-warning"),
            "Compléter le data catalogue",
        ]), close_button=True),
        dbc.ModalBody([
            html.P(
                "Renseignez tous les prérequis avant de lancer les calculs. "
                "Tous les champs sont obligatoires.",
                className="text-muted small",
            ),
            # ── Section 1 : Périmètre du rapport ──────────────────────
            html.Hr(),
            html.H6([html.I(className="fa fa-bullseye me-2 text-primary"),
                     "Périmètre du rapport"], className="text-secondary"),
            dbc.Label("Mode de rapport :", className="small fw-bold mt-2"),
            dbc.RadioItems(
                id="dc-report-mode",
                options=[
                    {"label": " Rapport complet (taux bruts + lissage + validation + benchmarking)",
                     "value": "full_report"},
                    {"label": " Taux bruts uniquement (sans lissage)",
                     "value": "raw_rates"},
                    {"label": " Description du portefeuille seule (pas de calcul de taux)",
                     "value": "description"},
                ],
                value="full_report",
                inline=False,
            ),
            dbc.Label("Segmentation par sexe :", className="small fw-bold mt-3"),
            dbc.RadioItems(
                id="dc-gender",
                options=[
                    {"label": " Table unisex (agrégée)",          "value": "unisex"},
                    {"label": " Tables séparées Hommes / Femmes", "value": "by_sex"},
                ],
                value="unisex", inline=True,
            ),
            dbc.Label("Générer un rapport PDF en fin de calcul ?",
                      className="small fw-bold mt-3"),
            dbc.RadioItems(
                id="dc-write",
                options=[
                    {"label": " Oui, générer le PDF + notebook", "value": "yes"},
                    {"label": " Non, juste les calculs",          "value": "no"},
                ],
                value="yes", inline=True,
            ),

            # ── Section 2 : Méthodes de calcul ────────────────────────
            html.Hr(),
            html.H6([html.I(className="fa fa-cogs me-2 text-primary"),
                     "Méthodes de calcul"], className="text-secondary"),
            dbc.Label("Choix des méthodes :", className="small fw-bold mt-2"),
            dbc.RadioItems(
                id="dc-methods-mode",
                options=[
                    {"label": " Mode auto (le système choisit les méthodes adaptées)",
                     "value": "auto"},
                    {"label": " Préciser chaque méthode manuellement",
                     "value": "explicit"},
                ],
                value="auto", inline=False,
            ),
            html.Div(id="dc-methods-explicit-container",
                     style={"display": "none"}, className="mt-2"),

            # ── Section 3 : Période d'observation ─────────────────────
            html.Hr(),
            html.H6([html.I(className="fa fa-calendar me-2 text-primary"),
                     "Période d'observation"], className="text-secondary"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Année de début (1er décès observé) :",
                              className="small fw-bold"),
                    dbc.Input(id="dc-start-year", type="number",
                              value=None, placeholder="ex : 1983",
                              size="sm"),
                ], md=4),
                dbc.Col([
                    dbc.Label("Année de fin (dernier décès observé) :",
                              className="small fw-bold"),
                    dbc.Input(id="dc-end-year", type="number",
                              value=None, placeholder="ex : 2010",
                              size="sm"),
                ], md=4),
            ], className="mt-2"),
        ]),
        dbc.ModalFooter([
            dbc.Button(
                [html.I(className="fa fa-check me-1"), "Confirmer"],
                id="btn-datacatalogue-confirm",
                color="primary", n_clicks=0,
            ),
            dbc.Button(
                "Annuler",
                id="btn-datacatalogue-cancel",
                color="secondary", outline=True, n_clicks=0,
                className="ms-2",
            ),
        ]),
    ], id="modal-datacatalogue", size="lg", is_open=False, scrollable=True)


def _disambiguation_modal() -> dbc.Modal:
    """
    Modal de désambiguation : tableau interactif de mapping colonnes +
    formulaire pour les prérequis manquants (table de référence, sexe, dates…).
    Contenu dynamique rendu par le callback render_disambiguation_modal().
    """
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle([
            html.I(className="fa fa-clipboard-check me-2 text-primary"),
            "Informations requises avant de lancer l'analyse",
        ]), close_button=True),
        dbc.ModalBody(html.Div(id="modal-disambiguation-body")),
        dbc.ModalFooter([
            dbc.Button(
                [html.I(className="fa fa-check me-1"), "Confirmer et lancer"],
                id="btn-disambiguation-confirm",
                color="primary",
                n_clicks=0,
            ),
            dbc.Button(
                "Annuler",
                id="btn-disambiguation-cancel",
                color="secondary",
                outline=True,
                n_clicks=0,
                className="ms-2",
            ),
        ]),
    ], id="modal-disambiguation", size="xl", is_open=False, scrollable=True)


def _render_column_mapping_table(
    df_columns: list[str],
    suggestion: dict[str, str | None],
) -> html.Div:
    """Tableau interactif mapping colonnes CSV ↔ champs actuariels."""
    from agents.master.disambiguation import EXPECTED_COLUMNS

    field_descriptions = {
        "date_naissance": "Date de naissance",
        "date_entree":    "Date d'entrée en observation",
        "date_sortie":    "Date de sortie d'observation",
        "cause_sortie":   "Cause de sortie (décès / autre)",
        "sexe":           "Sexe (optionnel)",
    }
    field_required = {
        "date_naissance": True,
        "date_entree":    True,
        "date_sortie":    True,
        "cause_sortie":   True,
        "sexe":           False,
    }

    options = [{"label": col, "value": col} for col in df_columns]
    options_with_none = [{"label": "— non disponible —", "value": ""}] + options

    rows = []
    for canonical, description in field_descriptions.items():
        suggested = suggestion.get(canonical) or ""
        required = field_required.get(canonical, True)
        badge = dbc.Badge("requis", color="danger", className="ms-1") if required \
                else dbc.Badge("optionnel", color="secondary", className="ms-1")
        confidence_color = "success" if suggested else "warning"
        confidence_icon = "fa-check-circle" if suggested else "fa-question-circle"
        rows.append(html.Tr([
            html.Td([
                html.Strong(description),
                badge,
                html.Br(),
                html.Small(canonical, className="text-muted font-monospace"),
            ], style={"verticalAlign": "middle", "width": "35%"}),
            html.Td(
                dbc.Select(
                    id={"type": "col-mapping-select", "field": canonical},
                    options=options if required else options_with_none,
                    value=suggested,
                    size="sm",
                ),
                style={"verticalAlign": "middle"},
            ),
            html.Td(
                html.I(
                    className=f"fa {confidence_icon} text-{confidence_color}",
                    title="Détecté automatiquement" if suggested else "Non détecté — à sélectionner",
                ),
                style={"verticalAlign": "middle", "textAlign": "center", "width": "5%"},
            ),
        ]))

    table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Champ actuariel"),
            html.Th(f"Colonne dans votre CSV ({len(df_columns)} colonnes)"),
            html.Th(""),
        ]), className="table-primary"),
        html.Tbody(rows),
    ], bordered=True, hover=True, size="sm", responsive=True)

    return html.Div([
        dbc.Alert([
            html.I(className="fa fa-info-circle me-2"),
            "Vérifiez la correspondance entre les colonnes de votre fichier et les champs actuariels.",
            html.Span(" Les colonnes en vert ont été détectées automatiquement.",
                      className="text-success"),
        ], color="info", className="mb-3 py-2"),
        table,
    ])


def _render_prerequisites_form(form_fields: list[dict]) -> html.Div:
    """Formulaire pour les prérequis non-mapping (dates, choix, entiers)."""
    if not form_fields:
        return html.Div()

    controls = []
    for field in form_fields:
        key         = field.get("key", "")
        label       = field.get("label", key)
        ftype       = field.get("type", "text")
        options     = field.get("options", [])
        placeholder = field.get("placeholder", "")
        description = field.get("description", "")
        default     = field.get("default", "")

        if ftype == "choice":
            control = dbc.Select(
                id={"type": "prereq-input", "key": key},
                options=[{"label": o, "value": o} for o in options],
                value=str(default) if default else options[0] if options else "",
                size="sm",
            )
        elif ftype in ("int", "float"):
            control = dbc.Input(
                id={"type": "prereq-input", "key": key},
                type="number",
                value=default if default else "",
                placeholder=placeholder or str(default),
                size="sm",
            )
        else:
            control = dbc.Input(
                id={"type": "prereq-input", "key": key},
                type="text",
                value=str(default) if default else "",
                placeholder=placeholder or label,
                size="sm",
            )

        controls.append(dbc.Row([
            dbc.Label(label, width=4, className="fw-bold small"),
            dbc.Col(control, width=8),
            dbc.Col(
                html.Small(description, className="text-muted"),
                width={"size": 8, "offset": 4},
            ) if description else None,
        ], className="mb-2 align-items-center"))

    return html.Div([
        html.Hr(className="my-3"),
        html.H6([html.I(className="fa fa-sliders-h me-2"), "Paramètres de l'étude"],
                className="text-secondary mb-3"),
        *controls,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

app.layout = dbc.Container([
    # Stores
    dcc.Store(id="store-page-load", data=True),   # détection rechargement page
    dcc.Store(id="_page-load-sink"),               # output fictif pour callback page-load
    dcc.Store(id="store-df-json"),
    dcc.Store(id="store-chat-history", data=[]),
    dcc.Store(id="store-last-event-idx", data=0),
    dcc.Store(id="store-pdf-path"),
    dcc.Store(id="store-txt-path"),
    dcc.Store(id="store-notebook-path"),
    dcc.Store(id="store-context-docs", data=[]),
    dcc.Store(id="store-step-mode", data=False),
    dcc.Store(id="store-agent-internals", data=[]),
    dcc.Store(id="store-disambiguation", data=None),  # données en attente de désambiguation
    # Trigger ouverture du modal Data Catalogue : poll_agent y écrit
    # {open: True, suggestions: {...}} quand event datacatalogue_incomplete arrive.
    dcc.Store(id="store-datacatalogue-trigger", data={}),
    # Bouton fantôme pour le callback toggle_disambiguation_modal
    # (rendu visible dans _chat_bubble, mais doit exister dans le layout statique)
    html.Button(id="btn-open-disambiguation", n_clicks=0, style={"display": "none"}),

    # Téléchargements
    dcc.Download(id="download-pdf"),
    dcc.Download(id="download-txt"),
    dcc.Download(id="download-notebook"),
    dcc.Download(id="download-rapport-from-list"),
    dcc.Interval(id="rapports-poll", interval=5000, n_intervals=0, disabled=False),

    # Polling interval (désactivé par défaut)
    dcc.Interval(id="interval-poll", interval=400, n_intervals=0, disabled=True),

    # Interval one-shot pour attacher l'écouteur Enter sur chat-input
    dcc.Interval(id="init-listeners", interval=600, n_intervals=0, max_intervals=1, disabled=False),

    # Modal désambiguation (mapping colonnes + formulaire prérequis)
    _disambiguation_modal(),
    _datacatalogue_modal(),

    # Header
    dbc.Navbar(
        dbc.Container([
            html.Span([
                html.I(className="fa fa-chart-line me-2 text-warning"),
                html.Strong("Agent Actuariat", className="text-white fs-5"),
            ]),
            html.Span("v2.0 — DEV", className="text-white-50 small"),
        ], fluid=True),
        color="dark", dark=True, className="mb-0",
    ),

    # Tabs principales
    dbc.Tabs([
        dbc.Tab(label="Rapport guidé", tab_id="tab-writer",
                children=_writer_tab()),
        dbc.Tab(label="DEV", tab_id="tab-dev",
                children=_dev_tab()),
        dbc.Tab(label="Doctrine RAG", tab_id="tab-doctrine",
                children=doctrine_ui.doctrine_tab()),
    ], id="main-tabs", active_tab="tab-writer"),

], fluid=True, className="px-0")

doctrine_ui.register_callbacks(app)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — CSV
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("restore-session-info", "children"),
    Output("store-chat-history", "data", allow_duplicate=True),
    Input("btn-restore-session", "n_clicks"),
    State("input-session-id", "value"),
    prevent_initial_call=True,
)
def cb_restore_session(n_clicks, session_id):
    if not session_id or not session_id.strip():
        raise PreventUpdate
    status_msg, restored_history = restore_session(session_id.strip())
    success = "introuvable" not in status_msg and "vide" not in status_msg and "corrompue" not in status_msg
    alert = dbc.Alert(
        [html.I(className=f"fa fa-{'check' if success else 'times'}-circle me-2"),
         status_msg],
        color="success" if success else "danger",
        className="mb-0 py-1 px-2",
    )
    return alert, (restored_history if success else [])


@app.callback(
    Output("_page-load-sink", "data"),
    Input("store-page-load", "data"),
    prevent_initial_call=False,
)
def _on_page_load(_):
    """
    Réinitialise le _writer_state à chaque chargement/rechargement de page.
    store-page-load est un dcc.Store(storage_type='memory') — il est remis à True
    par le navigateur à chaque refresh, ce qui déclenche ce callback.
    """
    with _writer_lock:
        _writer_state["session_id"]        = None
        _writer_state["data_store"]        = {}
        _writer_state["events"]            = []
        _writer_state["running"]           = False
        _writer_state["pending_tool_call"] = None
        _writer_state["context_docs"]      = []
    return dash.no_update


@app.callback(
    Output("store-df-json", "data"),
    Output("csv-info", "children"),
    Output("store-chat-history", "data", allow_duplicate=True),
    Output("store-last-event-idx", "data", allow_duplicate=True),
    Input("upload-csv", "contents"),
    State("upload-csv", "filename"),
    prevent_initial_call=True,
)
def upload_csv(contents, filename):
    if contents is None:
        raise PreventUpdate

    df, err = _parse_csv(contents, filename)
    if err:
        return None, dbc.Alert(err, color="danger", className="mb-0")

    df_json = df.to_json(orient="split")

    # Réinitialiser complètement le state pour la nouvelle session
    session_id = _new_session_id()
    with _writer_lock:
        _writer_state["data_store"]        = {}
        _writer_state["session_id"]        = session_id
        _writer_state["csv_filename"]      = filename
        _writer_state["events"]            = []
        _writer_state["running"]           = False
        _writer_state["pending_tool_call"] = None
        _writer_state["context_docs"]      = []

    caps = get_capabilities()
    report = build_mapping_report(df, caps)

    ready_fns = sum(1 for s in report["fn_readiness"].values() if s["ready"])
    total_fns = len(report["fn_readiness"])

    # ── Enregistrer le dataset (écriture unique) via MemoryManager ───────────
    from session.memory_manager import MemoryManager
    mm = MemoryManager(session_id)
    mm.load()
    mm.register_dataset(df, csv_filename=filename)
    # Propager le column_mapping AUTO-DÉTECTÉ dans le SessionState.
    # `column_mapping_confirmed` reste False : la validation du mapping est
    # désormais l'action explicite du bouton « Valider le mapping » (plan
    # bouton/clone). Tant qu'il n'est pas cliqué, le Master explore sur
    # l'original ; aucun calcul Builder/Writer n'est autorisé.
    mm.state.column_mapping           = report["matched"]
    mm.state.column_mapping_confirmed = False
    mm.state.column_mapping_unmatched = list(report["unmatched"].keys())
    mm.save()

    # ── Auto-détection période d'observation (pour pré-remplir le modal
    # « Compléter le data catalogue »). Min/max d'année de date_sortie chez
    # les décès observés. Best-effort : si le mapping rate ou les dates ne
    # parsent pas, on laisse None. Plan datacatalogue-gate 2026-05-25.
    auto_start_year: int | None = None
    auto_end_year: int | None = None
    _date_sortie_col = report["matched"].get("date_sortie")
    _cause_sortie_col = report["matched"].get("cause_sortie")
    if _date_sortie_col and _cause_sortie_col:
        try:
            from tools._shared.date_parsing import parse_dates_fr
            _parsed = parse_dates_fr(df[_date_sortie_col])
            _is_dead = df[_cause_sortie_col].astype(str).str.strip().str.lower(
                ).str.startswith(("dec", "déc", "d ", "1"))
            _years = _parsed[_is_dead].dt.year.dropna()
            if len(_years):
                auto_start_year = int(_years.min())
                auto_end_year = int(_years.max())
        except Exception:
            pass

    # Persister le mapping auto-détecté dans data_store (exploration Master).
    with _writer_lock:
        ds = _writer_state["data_store"]
        ds["_dataset_ref"]             = session_id
        ds["column_mapping"]           = report["matched"]
        ds["column_mapping_confirmed"] = False
        ds["column_mapping_unmatched"] = list(report["unmatched"].keys())
        # Suggestions pour pré-remplir le modal au moment d'ouverture
        ds["_auto_period"] = {
            "start_year": auto_start_year,
            "end_year":   auto_end_year,
        }
        # Pré-remplissage IMPLICITE de study_plan avec les années
        # auto-détectées : permet à la gate Builder de passer sans que
        # l'user ait à interagir avec la bulle (les valeurs restent
        # modifiables via le bouton sidebar). Plan datacatalogue-gate.
        if auto_start_year is not None and auto_end_year is not None:
            sp = ds.setdefault("study_plan", {})
            sp.setdefault("start_year", int(auto_start_year))
            sp.setdefault("end_year",   int(auto_end_year))
            sp.setdefault("observation_period_years",
                          [int(auto_start_year), int(auto_end_year)])
            sp.setdefault("num_observation_years",
                          int(auto_end_year) - int(auto_start_year) + 1)

    info = html.Div([
        dbc.Alert(
            [html.I(className="fa fa-check-circle me-2"),
             html.Strong(filename),
             f" — {len(df):,} lignes, {len(df.columns)} colonnes",
             html.Br(),
             f"{ready_fns}/{total_fns} fonctions disponibles"],
            color="success", className="mb-2 py-2",
        ),
        _mapping_badge(df),
        # Bouton de validation du mapping → crée la base de travail (clone
        # normalisé). Obligatoire avant tout calcul Builder/Writer.
        dbc.Button(
            [html.I(className="fa fa-check-double me-2"),
             "Valider le mapping et créer la base de travail"],
            id="btn-validate-mapping", color="primary", size="sm",
            className="mt-2 w-100",
        ),
        # Le bouton « Compléter le data catalogue » a été retiré : la bulle
        # inline dans le chat (rôle `_datacatalogue_form`) gère tout le flux.
        # On garde un bouton fantôme caché pour ne pas casser le callback
        # toggle_datacatalogue_modal qui le référence encore (sera nettoyé
        # avec le modal lui-même au prochain lot UI).
        html.Button(id="btn-open-datacatalogue", n_clicks=0,
                    style={"display": "none"}),
    ])
    return df_json, info, [], 0


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Documents de contexte
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text(content_b64: str, filename: str) -> str:
    """Extrait le texte d'un fichier uploadé (PDF, CSV, TXT)."""
    try:
        _, data = content_b64.split(",", 1)
        raw = base64.b64decode(data)
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=raw, filetype="pdf")
                text = "\n\n".join(page.get_text() for page in doc)
                doc.close()
                return text[:8000]  # tronqué pour le contexte LLM
            except Exception:
                return f"[PDF chargé : {filename} — extraction de texte non disponible]"
        else:
            # CSV, TXT, JSON — décode en texte
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    text = raw.decode(enc)
                    return text[:8000]
                except UnicodeDecodeError:
                    continue
            return f"[Fichier binaire non décodable : {filename}]"
    except Exception as exc:
        return f"[Erreur lors du chargement de {filename} : {exc}]"


@app.callback(
    Output("store-context-docs", "data"),
    Output("context-docs-list", "children"),
    Input("upload-context", "contents"),
    State("upload-context", "filename"),
    State("store-context-docs", "data"),
    prevent_initial_call=True,
)
def upload_context(contents_list, filenames, existing_docs):
    if not contents_list:
        raise PreventUpdate

    existing_docs = existing_docs or []
    existing_names = {d["name"] for d in existing_docs}

    for contents, filename in zip(contents_list, filenames):
        if filename in existing_names:
            continue  # ne pas dédupliquer
        text = _extract_text(contents, filename)
        existing_docs.append({"name": filename, "content": text})
        existing_names.add(filename)

    # Mettre à jour le context_docs dans le thread state
    with _writer_lock:
        _writer_state["context_docs"] = list(existing_docs)

    # Affichage de la liste
    items = [
        dbc.ListGroupItem(
            [
                html.I(className="fa fa-file-alt me-2 text-muted"),
                html.Span(d["name"], className="small"),
                html.Span(
                    f" ({len(d['content'])} car.)",
                    className="small text-muted ms-1",
                ),
            ],
            className="py-1 px-2",
        )
        for d in existing_docs
    ]
    badge = dbc.Badge(f"{len(existing_docs)} doc(s)", color="info", className="me-1")
    return existing_docs, html.Div([
        badge,
        dbc.ListGroup(items, flush=True, className="mt-1 small"),
    ]) if items else html.Div()


# ─────────────────────────────────────────────────────────────────────────────
# Callback — Document ajouté en cours de conversation
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("store-chat-history", "data", allow_duplicate=True),
    Output("interval-poll", "disabled", allow_duplicate=True),
    Output("agent-status-badge", "children", allow_duplicate=True),
    Output("agent-status-badge", "color", allow_duplicate=True),
    Output("store-last-event-idx", "data", allow_duplicate=True),
    Output("mid-chat-doc-name", "children"),
    Input("upload-mid-chat", "contents"),
    State("upload-mid-chat", "filename"),
    State("store-chat-history", "data"),
    State("store-df-json", "data"),
    State("store-last-event-idx", "data"),
    State("switch-step-mode", "value"),
    prevent_initial_call=True,
)
def handle_mid_chat_upload(contents, filename, history, df_json, _last_idx, step_mode):
    if not contents or not filename:
        raise PreventUpdate

    # Extraire le texte
    text = _extract_text(contents, filename)

    # Ajouter aux context_docs
    new_doc = {"name": filename, "content": text}
    with _writer_lock:
        docs = _writer_state.get("context_docs") or []
        if not any(d["name"] == filename for d in docs):
            docs.append(new_doc)
        _writer_state["context_docs"] = docs

    # Message automatique — l'agent décide quoi faire avec le doc
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in ("csv",):
        auto_msg = (
            f"J'ai chargé un nouveau fichier de données : **{filename}**. "
            "Analyse-le et dis-moi ce que tu peux en faire pour notre étude."
        )
    else:
        auto_msg = (
            f"J'ai ajouté le document **{filename}** au contexte. "
            "Prends-en connaissance et explique brièvement comment tu comptes l'utiliser "
            "ou demande-moi des précisions si nécessaire."
        )

    history = list(history or [])
    history.append({"role": "user", "content": auto_msg})

    # Réinitialiser le state et lancer l'agent
    with _writer_lock:
        _writer_state["events"]           = []
        _writer_state["running"]          = True
        _writer_state["step_by_step"]     = bool(step_mode)
        _writer_state["pending_tool_call"] = None

    t = threading.Thread(
        target=_run_writer_in_thread,
        args=(history, df_json),
        daemon=True,
    )
    t.start()

    return history, False, "En cours…", "warning", 0, f"✓ {filename}"


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Chat
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("store-chat-history", "data"),
    Output("chat-input", "value"),
    Output("interval-poll", "disabled"),
    Output("agent-status-badge", "children"),
    Output("agent-status-badge", "color"),
    Output("store-last-event-idx", "data"),
    Input("btn-send", "n_clicks"),
    State("chat-input", "value"),
    State("store-chat-history", "data"),
    State("store-df-json", "data"),
    State("store-last-event-idx", "data"),
    State("switch-step-mode", "value"),
    prevent_initial_call=True,
)
def send_message(n_clicks, message, history, df_json, _last_idx, step_mode):
    if not n_clicks or not message or not message.strip():
        raise PreventUpdate

    history = history or []
    history.append({"role": "user", "content": message.strip()})

    # Réinitialiser le state
    with _writer_lock:
        _writer_state["events"] = []
        _writer_state["running"] = True
        _writer_state["step_by_step"] = bool(step_mode)
        _writer_state["pending_tool_call"] = None

    # Lancer le thread
    t = threading.Thread(
        target=_run_writer_in_thread,
        args=(history, df_json),
        daemon=True,
    )
    t.start()

    return history, "", False, "En cours…", "warning", 0


_AGENT_COLORS = {
    "MasterAgent":  "info",
    "BuilderAgent": "warning",
    "WriterAgent":  "primary",
}

_INTERNALS_COLORS = {
    "new_turn":     "#888888",
    "agent_switch": "#569CD6",
    "master_stage": "#FFD580",   # ambre clair — pour repérer le tracé Master
    "llm_input":    "#9CDCFE",
    "llm_output":   "#B5CEA8",
    "tool_call":    "#DCDCAA",
    "tool_result":  "#4EC9B0",
    "message":      "#CE9178",
    "error":        "#F44747",
    "done":         "#608B4E",
    "awaiting_approval": "#C586C0",
}

_INTERNALS_ICONS = {
    "new_turn":     "─────",
    "agent_switch": "▶",
    "master_stage": "📍",
    "llm_input":    "→",
    "llm_output":   "←",
    "tool_call":    "🔧",
    "tool_result":  "✅",
    "message":      "💬",
    "error":        "❌",
    "done":         "✓",
    "awaiting_approval": "⏸",
}


def _internals_entry(ev: dict) -> html.Div:
    """Construit une ou plusieurs lignes de log pour le panneau internals."""
    ev_type = ev.get("type", "")
    color   = _INTERNALS_COLORS.get(ev_type, "#D4D4D4")
    icon    = _INTERNALS_ICONS.get(ev_type, "·")

    if ev_type == "new_turn":
        user_msg = (ev.get("user_msg") or "")[:80]
        lines = [
            html.Div(
                f"───── Nouveau tour ─────  {user_msg}",
                style={"color": "#666666", "marginTop": "8px", "marginBottom": "4px",
                       "borderTop": "1px solid #333333", "paddingTop": "4px"},
            )
        ]
        return html.Div(lines)

    elif ev_type == "agent_switch":
        text = f"{icon} {ev.get('agent', '')} actif"

    elif ev_type == "agent_transition":
        # HOTFIX-pre-refacto-2026-05 (Bug 18, A3) — trigger générique.
        _frm = ev.get("from") or "?"
        _to  = ev.get("to") or "?"
        _rsn = ev.get("reason") or ""
        _txt = f"📍 [Switch model] {_frm} → {_to}"
        if _rsn:
            _txt += f" — {_rsn}"
        return html.Div(
            _txt,
            style={
                "color":        "#C8A24A",
                "marginBottom": "2px",
                "marginTop":    "2px",
                "paddingLeft":  "12px",
                "fontStyle":    "italic",
                "fontWeight":   "600",
            },
        )

    elif ev_type == "master_stage":
        stage = ev.get("stage", "?")
        label = ev.get("label", "")
        return html.Div(
            f"{icon} [Master {stage}] {label}",
            style={
                "color":         color,
                "marginBottom":  "2px",
                "marginTop":     "2px",
                "paddingLeft":   "12px",
                "fontStyle":     "italic",
                "fontWeight":    "500",
            },
        )

    elif ev_type == "llm_input":
        agent  = ev.get("agent", "")
        n_msg  = ev.get("n_messages", "?")
        mt     = ev.get("max_tokens", "?")
        tools  = "oui" if ev.get("has_tools") else "non"
        last_u = (ev.get("last_user") or "")[:200]
        sys_h  = (ev.get("system_head") or "")[:200]
        lines = [
            html.Div(
                f"{icon} [{agent}] → GPT-4o  |  {n_msg} messages  |  max_tokens={mt}  |  tools={tools}",
                style={"color": color, "marginBottom": "1px"},
            ),
            html.Div(
                f"   system: {sys_h}…",
                style={"color": "#7A7A7A", "marginBottom": "1px", "paddingLeft": "12px"},
            ),
            html.Div(
                f"   user:   {last_u}",
                style={"color": "#7A7A7A", "marginBottom": "4px", "paddingLeft": "12px"},
            ),
        ]
        return html.Div(lines)

    elif ev_type == "llm_output":
        agent   = ev.get("agent", "")
        reason  = ev.get("finish_reason", "?")
        pt      = ev.get("prompt_tokens")
        ct      = ev.get("completion_tokens")
        tt      = ev.get("total_tokens")
        ntc     = ev.get("n_tool_calls", 0)
        preview = (ev.get("content_preview") or "")[:300]
        token_str = f"{pt}+{ct}={tt} tokens" if tt else "tokens: ?"
        lines = [
            html.Div(
                f"{icon} [{agent}] ← GPT-4o  |  finish={reason}  |  {token_str}"
                + (f"  |  {ntc} tool_calls" if ntc else ""),
                style={"color": color, "marginBottom": "1px"},
            ),
        ]
        if preview:
            lines.append(html.Div(
                f"   {preview}",
                style={"color": "#A8A8A8", "marginBottom": "4px", "paddingLeft": "12px"},
            ))
        return html.Div(lines)

    elif ev_type == "tool_call":
        text = f"{icon} {ev.get('tool', '')}.{ev.get('function_name', '')}"
        p = ev.get("params") or {}
        if p:
            param_str = ", ".join(f"{k}={v}" for k, v in list(p.items())[:4])
            text += f"({param_str})"

    elif ev_type == "tool_result":
        result = ev.get("result") or {}
        if "erreur" in result:
            text = f"❌ Erreur : {str(result['erreur'])[:120]}"
        else:
            keys = [k for k in result if k not in ("erreur", "image_b64", "samples", "table", "columns_header")]
            text = f"{icon} {ev.get('function_name', '')} → {', '.join(keys[:6])}"

    elif ev_type == "message":
        content = (ev.get("content") or "")[:300]
        text = f"{icon} {content}"

    elif ev_type == "error":
        text = f"{icon} {(ev.get('message') or '')[:200]}"

    elif ev_type == "done":
        text = f"{icon} Tour terminé"

    else:
        text = f"· [{ev_type}]"

    return html.Div(text, style={"color": color, "marginBottom": "3px", "lineHeight": "1.4"})


def _derive_dc_state(data_store: dict | None,
                      history: list[dict] | None) -> tuple[str, str]:
    """Retourne (className, label) pour l'icône datacatalogue sidebar.

    Règles (priorité du haut vers le bas) :
      1. Bulle inline ouverte (entry `_datacatalogue_form` non soumise)
         → « à compléter » 🟡
      2. compute_datacatalogue_state(ds).complete == True
         → « prêt » 🟢
      3. Sinon → « pas requis » ⚪

    Plan refonte garde-fou 2026-06-03 (Partie A).
    """
    has_open_bubble = any(
        h.get("role") == "_datacatalogue_form" and not h.get("submitted")
        for h in (history or [])
    )
    if has_open_bubble:
        return ("fa fa-circle-exclamation text-warning me-2",
                " · à compléter")
    try:
        from knowledge_base.report_template.datacatalogue import (
            compute_datacatalogue_state,
        )
        dc = compute_datacatalogue_state(data_store or {})
        if dc.complete:
            return ("fa fa-circle-check text-success me-2",
                    " · prêt")
    except Exception:
        pass
    return ("fa fa-circle-question text-muted me-2",
            " · pas requis")


@app.callback(
    Output("store-chat-history", "data", allow_duplicate=True),
    Input("dc-status-row", "n_clicks"),
    State("store-chat-history", "data"),
    prevent_initial_call=True,
)
def open_dc_bubble_from_icon(n_clicks, history):
    """Au clic sur l'icône sidebar « Data catalogue », ouvre la bulle
    inline du formulaire (si elle n'est pas déjà ouverte). Permet à
    l'utilisateur de réviser/modifier sans attendre que l'agent le
    demande. Plan refonte garde-fou 2026-06-03 (Partie A.4)."""
    if not n_clicks:
        raise PreventUpdate
    history = list(history or [])
    if any(h.get("role") == "_datacatalogue_form" and not h.get("submitted")
           for h in history):
        raise PreventUpdate
    with _writer_lock:
        ds = dict(_writer_state.get("data_store") or {})
    auto = (ds.get("_auto_period") or {})
    try:
        from knowledge_base.report_template.datacatalogue import (
            compute_datacatalogue_state,
        )
        missing = compute_datacatalogue_state(ds).missing
    except Exception:
        missing = []
    import time as _time
    history.append({
        "role":        "_datacatalogue_form",
        "form_id":     f"dc-{int(_time.time() * 1000)}",
        "missing":     missing,
        "suggestions": auto,
        "submitted":   False,
    })
    return history


@app.callback(
    Output("dc-icon", "className"),
    Output("dc-status-label", "children"),
    Input("store-chat-history", "data"),
    Input("interval-poll", "n_intervals"),
    prevent_initial_call=False,
)
def refresh_dc_icon(history, _n):
    """Rafraîchit l'icône datacatalogue à chaque changement de history
    (bulle créée / soumise) et à chaque tick de polling agent.
    Plan refonte garde-fou 2026-06-03 (Partie A)."""
    with _writer_lock:
        ds = dict(_writer_state.get("data_store") or {})
    return _derive_dc_state(ds, history)


@app.callback(
    Output("chat-messages", "children"),
    Output("interval-poll", "disabled", allow_duplicate=True),
    Output("agent-status-badge", "children", allow_duplicate=True),
    Output("agent-status-badge", "color", allow_duplicate=True),
    Output("store-chat-history", "data", allow_duplicate=True),
    Output("store-last-event-idx", "data", allow_duplicate=True),
    Output("store-pdf-path",      "data", allow_duplicate=True),
    Output("store-txt-path",      "data", allow_duplicate=True),
    Output("store-notebook-path", "data", allow_duplicate=True),
    Output("step-approval-banner", "children", allow_duplicate=True),
    Output("agent-internals-log", "children", allow_duplicate=True),
    Output("internals-agent-badge", "children", allow_duplicate=True),
    Output("store-disambiguation", "data", allow_duplicate=True),
    Output("store-datacatalogue-trigger", "data", allow_duplicate=True),
    Input("interval-poll", "n_intervals"),
    State("store-chat-history", "data"),
    State("store-last-event-idx", "data"),
    State("agent-internals-log", "children"),
    prevent_initial_call=True,
)
def poll_agent(n_intervals, history, last_idx, existing_internals):
    with _writer_lock:
        events = list(_writer_state["events"])
        running = _writer_state["running"]

    history = list(history or [])
    new_events = events[last_idx:]

    # ── Skip re-render quand RIEN n'a changé ─────────────────────────────────
    # Sans ça, poll_agent reconstruit chat-messages toutes les 400 ms : les
    # composants Dash (RadioItems, Input) de la bulle inline du data
    # catalogue sont re-rendus avec leur `value` initial → les sélections
    # de l'utilisateur sont écrasées à chaque tick. En l'absence d'events
    # nouveaux on retourne dash.no_update pour les outputs lourds, en
    # préservant uniquement les badges et le flag poll_disabled.
    if not new_events:
        done = not running
        poll_disabled = done
        if done:
            status_text, status_color = "Prêt", "success"
        else:
            status_text, status_color = "En cours…", "warning"
        return (
            dash.no_update,                         # chat-messages
            poll_disabled, status_text, status_color,
            dash.no_update,                         # store-chat-history
            dash.no_update,                         # store-last-event-idx
            dash.no_update, dash.no_update,         # pdf/txt paths
            dash.no_update,                         # notebook path
            dash.no_update,                         # step-approval-banner
            dash.no_update,                         # agent-internals-log
            dash.no_update,                         # internals-agent-badge
            dash.no_update,                         # store-disambiguation
            dash.no_update,                         # store-datacatalogue-trigger
        )

    pdf_path = txt_path = notebook_path = None
    disambiguation_data = dash.no_update
    datacatalogue_trigger = dash.no_update

    # Badge agent courant (dernier agent_switch vu)
    current_agent = None

    # Accumuler les nouvelles entrées internals
    new_internals = list(existing_internals or [])

    # On capture aussi la dernière stage rencontrée pour enrichir le badge
    # « En cours » du chat (sinon on voit juste « MasterAgent » sans contexte).
    current_stage_label: str | None = None

    for ev in new_events:
        ev_type = ev.get("type")

        # Panneau internals : toujours loguer
        new_internals.append(_internals_entry(ev))

        if ev_type == "agent_switch":
            current_agent = ev.get("agent")
            current_stage_label = None  # reset au changement d'agent

        elif ev_type == "master_stage":
            # Garder le label de la dernière stage pour l'afficher en badge
            current_stage_label = ev.get("label") or current_stage_label

        elif ev_type == "message":
            content = ev.get("content", "")
            history.append({"role": "assistant", "content": content})
            # Extraire le chemin PDF depuis <WRITE_DONE: /path/to/file.pdf>
            import re as _re
            _wd = _re.search(r'<WRITE_DONE[:\s]+([^\s>]+\.pdf)', content)
            if _wd:
                pdf_path = _wd.group(1)

        elif ev_type == "tool_call":
            # Tracé internals + bulle chat + mise à jour badge contexte
            _tn = f"{ev.get('tool', '')}.{ev.get('function_name', '')}"
            current_stage_label = f"appel {_tn}"
            history.append({
                "role": "_tool_call",
                "tool": ev.get("tool", ""),
                "function_name": ev.get("function_name", ""),
                "content": f"Appel : {ev.get('tool')}.{ev.get('function_name')}",
            })

        elif ev_type == "tool_result":
            result = ev.get("result", {})
            image_b64      = result.get("image_b64")
            samples        = result.get("samples")
            table          = result.get("table")
            columns_header = result.get("columns_header")
            result_keys = [
                k for k in result
                if k not in ("erreur", "image_b64", "samples", "n_samples",
                             "table", "columns_header")
            ]
            history.append({
                "role":           "_tool_result",
                "function_name":  ev.get("function_name", ""),
                "image_b64":      image_b64,
                "samples":        samples,
                "table":          table,
                "columns_header": columns_header,
                "result_keys":    result_keys,
                "content":        "",
            })
            # Détecter fichiers générés
            out_path = str(result.get("output_path", ""))
            if out_path and result.get("succes"):
                if out_path.endswith(".pdf"):
                    pdf_path = out_path
                elif out_path.endswith(".txt"):
                    txt_path = out_path
                elif out_path.endswith(".ipynb"):
                    notebook_path = out_path

        elif ev_type == "report_ready":
            # Rapport PDF généré par le WriterAgent pipeline
            out_path = str(ev.get("output_path", ""))
            if out_path and out_path.endswith(".pdf"):
                pdf_path = out_path

        elif ev_type == "disambiguation_required":
            # Ouvrir le modal de désambiguation
            disambiguation_data = ev
            # Ne pas stocker ev dans history (non sérialisable proprement)
            history.append({
                "role":    "_disambiguation",
                "content": "Informations requises avant de lancer l'analyse.",
            })

        elif ev_type == "datacatalogue_incomplete":
            # La gate Builder refuse : on injecte une BULLE INLINE dans le
            # chat (alternative MCP-UI au modal popup). L'utilisateur remplit
            # le formulaire sans quitter la conversation. Les suggestions
            # (années auto-détectées à l'upload) sont pré-remplies. Un seul
            # form actif à la fois — on ignore si une bulle non-soumise est
            # déjà présente dans history.
            with _writer_lock:
                _auto = (_writer_state["data_store"] or {}).get("_auto_period") or {}
            _already_pending = any(
                h.get("role") == "_datacatalogue_form" and not h.get("submitted")
                for h in history
            )
            if not _already_pending:
                import time as _time
                _form_id = f"dc-{int(_time.time() * 1000)}"
                history.append({
                    "role":        "_datacatalogue_form",
                    "form_id":     _form_id,
                    "missing":     ev.get("missing") or [],
                    "suggestions": _auto,
                    "submitted":   False,
                })
                # Toujours révéler le bouton sidebar (accès manuel rapide)
                datacatalogue_trigger = {
                    "ts":          n_intervals,
                    "reveal":      True,
                }

        elif ev_type == "decision_required":
            # Bulle inline « décision tool » (ex. smoothing : 8 violations
            # de monotonie → user choisit lambda × 2 / autre méthode /
            # accepter). Plan refonte garde-fou 2026-06-03.
            _already_dr = any(
                h.get("role") == "_decision_required_panel"
                and not h.get("submitted")
                for h in history
            )
            if not _already_dr:
                import time as _time
                history.append({
                    "role":      "_decision_required_panel",
                    "form_id":   f"dr-{int(_time.time() * 1000)}",
                    "tool":      ev.get("tool", ""),
                    "reason":    ev.get("reason", ""),
                    "options":   ev.get("options", []),
                    "submitted": False,
                })

        elif ev_type == "error":
            history.append({"role": "assistant", "content": f"⚠️ Erreur : {ev.get('message', '')}"})

    new_idx = len(events)

    # Construire les bulles chat
    bubbles = []
    for h in history:
        role = h.get("role", "user")
        content = h.get("content", "")
        if role == "user":
            bubbles.append(_chat_bubble("user", content))
        elif role == "assistant":
            bubbles.append(_chat_bubble("assistant", content))
        elif role == "_tool_call":
            bubbles.append(_chat_bubble("assistant", content, extra={
                "type": "tool_call",
                "tool": h.get("tool", ""),
                "function_name": h.get("function_name", ""),
            }))
        elif role == "_tool_result":
            img            = h.get("image_b64")
            samples        = h.get("samples")
            table          = h.get("table")
            columns_header = h.get("columns_header")
            if table:
                bubbles.append(_chat_bubble("assistant", "", extra={
                    "type":           "tool_result",
                    "table":          table,
                    "columns_header": columns_header,
                    "function_name":  h.get("function_name", ""),
                }))
            elif samples:
                bubbles.append(_chat_bubble("assistant", "", extra={
                    "type":          "tool_result",
                    "samples":       samples,
                    "function_name": h.get("function_name", ""),
                }))
            elif img:
                bubbles.append(_chat_bubble("assistant", "", extra={
                    "type":          "tool_result",
                    "image_b64":     img,
                    "function_name": h.get("function_name", ""),
                }))
            elif h.get("result_keys"):
                bubbles.append(_chat_bubble("assistant", "", extra={
                    "type":          "tool_result",
                    "function_name": h.get("function_name", ""),
                    "result_keys":   h.get("result_keys", []),
                }))
        elif role == "_disambiguation":
            bubbles.append(_chat_bubble("_disambiguation", content))
        elif role == "_datacatalogue_form":
            bubbles.append(_render_datacatalogue_bubble(h))
        elif role == "_decision_required_panel":
            bubbles.append(_render_decision_required_bubble(h))

    done = not running
    poll_disabled = done

    # Badge principal : agent actif + contexte court (label de la dernière
    # stage ou tool en cours), pour donner un signal de vie plus parlant
    # que le simple nom de l'agent.
    if done:
        status_text  = "Prêt"
        status_color = "success"
    elif current_agent:
        if current_stage_label:
            # Tronquer pour rester lisible sur le badge
            _ctx = current_stage_label
            if len(_ctx) > 60:
                _ctx = _ctx[:57] + "…"
            status_text  = f"{current_agent} · {_ctx}"
        else:
            status_text  = current_agent
        status_color = _AGENT_COLORS.get(current_agent, "warning")
    else:
        status_text  = "En cours…"
        status_color = "warning"

    # Badge internals
    internals_badge = current_agent or ("Terminé" if done else "—")

    # Bannière mode pas à pas
    with _writer_lock:
        pending = _writer_state.get("pending_tool_call")
    banner = _pending_banner(pending) if pending else html.Div()

    return (bubbles, poll_disabled, status_text, status_color,
            history, new_idx, pdf_path, txt_path, notebook_path,
            banner, new_internals, internals_badge, disambiguation_data,
            datacatalogue_trigger)


# ─────────────────────────────────────────────────────────────────────────────
# Clientside — Touche Entrée pour envoyer le message
# ─────────────────────────────────────────────────────────────────────────────

app.clientside_callback(
    """
    function(_) {
        var el = document.getElementById('chat-input');
        if (el && !el._enterBound) {
            el._enterBound = true;
            el.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    var btn = document.getElementById('btn-send');
                    if (btn) btn.click();
                }
            });
        }
        return true;
    }
    """,
    Output("init-listeners", "disabled"),
    Input("init-listeners", "n_intervals"),
    prevent_initial_call=False,
)




# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Mode pas à pas (Approuver / Annuler)
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("step-approval-banner", "children", allow_duplicate=True),
    Input("btn-step-approve", "n_clicks"),
    prevent_initial_call=True,
)
def approve_step(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    with _writer_lock:
        _writer_state["pending_tool_call"] = None
    _step_approval_event.set()
    return html.Div()


@app.callback(
    Output("step-approval-banner", "children", allow_duplicate=True),
    Input("btn-step-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_step(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    _step_cancel_flag[0] = True
    with _writer_lock:
        _writer_state["pending_tool_call"] = None
    _step_approval_event.set()
    return html.Div()


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Téléchargement PDF
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("download-pdf", "data"),
    Input("store-pdf-path", "data"),
    prevent_initial_call=True,
)
def trigger_pdf_download(pdf_path):
    if not pdf_path:
        raise PreventUpdate
    from pathlib import Path as _Path
    p = _Path(pdf_path)
    if not p.exists():
        raise PreventUpdate
    return dcc.send_file(str(p))


@app.callback(
    Output("download-txt", "data"),
    Input("store-txt-path", "data"),
    prevent_initial_call=True,
)
def trigger_txt_download(txt_path):
    if not txt_path:
        raise PreventUpdate
    from pathlib import Path as _Path
    p = _Path(txt_path)
    if not p.exists():
        raise PreventUpdate
    return dcc.send_file(str(p))


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Section Rapports (sidebar gauche)
# ─────────────────────────────────────────────────────────────────────────────

_RAPPORTS_DIR = Path(__file__).resolve().parent / "session" / "rapports"


def _human_size(n: int) -> str:
    for unit in ("o", "Ko", "Mo", "Go"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "o" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} To"


@app.callback(
    Output("rapports-list", "children"),
    Input("btn-rapports-refresh", "n_clicks"),
    Input("rapports-poll", "n_intervals"),
    Input("store-pdf-path", "data"),
)
def render_rapports_list(_n_clicks, _n_intervals, _pdf_path):
    """Affiche la liste des PDFs du dossier session/rapports/, triés par
    date décroissante (plus récent en haut). Mise à jour automatique
    toutes les 5s + immédiate quand un nouveau PDF arrive."""
    from datetime import datetime as _dt2
    if not _RAPPORTS_DIR.exists():
        return html.Div("Aucun rapport pour l'instant.",
                        className="text-muted small fst-italic")
    pdfs = sorted(_RAPPORTS_DIR.glob("Rapport_*.pdf"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not pdfs:
        return html.Div("Aucun rapport pour l'instant.",
                        className="text-muted small fst-italic")

    items = []
    for pdf in pdfs[:30]:  # max 30 affichés (FIFO si plus)
        nb = pdf.with_suffix(".ipynb")
        size = _human_size(pdf.stat().st_size)
        mtime = _dt2.fromtimestamp(pdf.stat().st_mtime).strftime("%d/%m %H:%M")
        items.append(dbc.ListGroupItem([
            html.Div([
                html.I(className="fa fa-file-pdf me-2 text-danger"),
                html.Strong(pdf.stem.replace("Rapport_", "N° "),
                            className="small"),
                html.Span(f"  · {size} · {mtime}",
                          className="text-muted small ms-1"),
            ], className="mb-1"),
            dbc.ButtonGroup([
                dbc.Button(
                    [html.I(className="fa fa-download me-1"), "PDF"],
                    id={"type": "btn-rapport-dl", "kind": "pdf",
                        "name": pdf.name},
                    color="primary", size="sm", outline=True,
                ),
                dbc.Button(
                    [html.I(className="fa fa-download me-1"), "Notebook"],
                    id={"type": "btn-rapport-dl", "kind": "ipynb",
                        "name": nb.name},
                    color="secondary", size="sm", outline=True,
                    disabled=not nb.exists(),
                ),
            ], size="sm"),
        ], className="py-2"))
    return dbc.ListGroup(items, flush=True)


@app.callback(
    Output("download-rapport-from-list", "data"),
    Input({"type": "btn-rapport-dl", "kind": ALL, "name": ALL}, "n_clicks"),
    State({"type": "btn-rapport-dl", "kind": ALL, "name": ALL}, "id"),
    prevent_initial_call=True,
)
def trigger_rapport_download(n_clicks_list, ids_list):
    """Quand on clique un bouton de la liste, télécharge le fichier
    correspondant. Identifie quel bouton via callback_context."""
    ctx = callback_context
    if not ctx.triggered or not any(n_clicks_list or []):
        raise PreventUpdate
    # Identifier l'index cliqué via la valeur de n_clicks (le plus récent
    # est le seul à avoir un n_clicks > 0 sur ce cycle).
    triggered_prop = ctx.triggered[0]["prop_id"]
    if not triggered_prop or triggered_prop == ".":
        raise PreventUpdate
    import json as _json
    # prop_id ressemble à : '{"kind":"pdf","name":"Rapport_X.pdf","type":"btn-rapport-dl"}.n_clicks'
    id_part = triggered_prop.rsplit(".", 1)[0]
    try:
        id_dict = _json.loads(id_part)
    except Exception:
        raise PreventUpdate
    name = id_dict.get("name", "")
    if not name:
        raise PreventUpdate
    target = _RAPPORTS_DIR / name
    if not target.exists():
        raise PreventUpdate
    return dcc.send_file(str(target))


@app.callback(
    Output("download-notebook", "data"),
    Input("store-notebook-path", "data"),
    prevent_initial_call=True,
)
def trigger_notebook_download(nb_path):
    if not nb_path:
        raise PreventUpdate
    from pathlib import Path as _Path
    p = _Path(nb_path)
    if not p.exists():
        raise PreventUpdate
    return dcc.send_file(str(p))


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — DEV tab
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("dev-caps-panel", "children"),
    Input("btn-refresh-caps", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_caps(_):
    return _build_capability_cards()


@app.callback(
    Output("main-tabs", "active_tab"),
    Output("dev-tabs", "active_tab"),
    Output("dev-file-path-display", "value", allow_duplicate=True),
    Output("dev-code-editor", "value", allow_duplicate=True),
    Input({"type": "dev-view-code-btn", "tool": ALL, "fn": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def view_code_from_cap(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered or all((n or 0) == 0 for n in n_clicks_list):
        raise PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
        id_dict = json.loads(triggered_id)
    except Exception:
        raise PreventUpdate

    tool = id_dict.get("tool", "")
    fn = id_dict.get("fn", "")
    tools_root = Path(__file__).parent / "tools"
    file_path = tools_root / tool / f"{fn}.py"

    if not file_path.exists():
        raise PreventUpdate

    code = file_path.read_text(encoding="utf-8")
    return "tab-dev", "dev-code", str(file_path), code


@app.callback(
    Output("dev-code-editor", "value", allow_duplicate=True),
    Output("dev-file-path-display", "value", allow_duplicate=True),
    Input({"type": "dev-file-btn", "path": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def load_file_from_tree(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered or all((n or 0) == 0 for n in n_clicks_list):
        raise PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
        id_dict = json.loads(triggered_id)
    except Exception:
        raise PreventUpdate

    file_path = Path(id_dict.get("path", ""))
    if not file_path.exists():
        raise PreventUpdate

    code = file_path.read_text(encoding="utf-8")
    return code, str(file_path)


@app.callback(
    Output("dev-save-feedback", "children"),
    Input("btn-save-code", "n_clicks"),
    State("dev-file-path-display", "value"),
    State("dev-code-editor", "value"),
    prevent_initial_call=True,
)
def save_code(n_clicks, file_path, code):
    if not n_clicks or not file_path or not code:
        raise PreventUpdate
    try:
        Path(file_path).write_text(code, encoding="utf-8")
        return html.Span([html.I(className="fa fa-check-circle text-success me-1"),
                          f"Sauvegardé : {Path(file_path).name}"])
    except Exception as exc:
        return html.Span([html.I(className="fa fa-times-circle text-danger me-1"),
                          f"Erreur : {exc}"])


@app.callback(
    Output("modal-new-fn", "is_open"),
    Output("new-fn-tool", "value"),
    Output("new-fn-name", "value"),
    Output("new-fn-code", "value"),
    Input({"type": "dev-add-fn-btn", "tool": ALL}, "n_clicks"),
    Input("btn-new-fn-cancel", "n_clicks"),
    Input("btn-new-fn-create", "n_clicks"),
    State("new-fn-name", "value"),
    State("new-fn-desc", "value"),
    State("new-fn-req-cols", "value"),
    State("new-fn-opt-cols", "value"),
    State("new-fn-params", "value"),
    State("new-fn-code", "value"),
    State("new-fn-tool", "value"),
    prevent_initial_call=True,
)
def handle_new_fn_modal(add_clicks, cancel_clicks, create_clicks,
                        fn_name, fn_desc, req_cols, opt_cols, params_json, code, tool_name):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered = ctx.triggered[0]["prop_id"].split(".")[0]

    # Fermeture
    if "btn-new-fn-cancel" in triggered:
        return False, dash.no_update, "", ""

    # Création
    if "btn-new-fn-create" in triggered:
        if not fn_name or not tool_name or not code:
            return True, dash.no_update, dash.no_update, dash.no_update

        tools_root = Path(__file__).parent / "tools"
        target_dir = tools_root / tool_name
        target_dir.mkdir(parents=True, exist_ok=True)
        py_path = target_dir / f"{fn_name}.py"
        py_path.write_text(code, encoding="utf-8")

        # Le catalogue est maintenant dynamique (catalogue.py) — invalider le cache
        # pour que get_capabilities() reparse le nouveau fichier au prochain appel
        try:
            from tools.tool_registry import invalidate_capabilities_cache
            invalidate_capabilities_cache()
        except Exception:
            pass

        return False, dash.no_update, "", ""

    # Ouverture : décoder le tool depuis le bouton cliqué
    try:
        id_dict = json.loads(triggered)
        clicked_tool = id_dict.get("tool", "")
    except Exception:
        raise PreventUpdate

    template = _generate_fn_template(fn_name or "ma_fonction", fn_desc or "", req_cols or [], opt_cols or [])
    return True, clicked_tool, "", template


@app.callback(
    Output("new-fn-code", "value", allow_duplicate=True),
    Input("new-fn-name", "value"),
    Input("new-fn-req-cols", "value"),
    Input("new-fn-opt-cols", "value"),
    Input("new-fn-desc", "value"),
    prevent_initial_call=True,
)
def regenerate_fn_template(fn_name, req_cols, opt_cols, fn_desc):
    if not fn_name:
        raise PreventUpdate
    return _generate_fn_template(fn_name, fn_desc or "", req_cols or [], opt_cols or [])


def _generate_fn_template(fn_name: str, description: str,
                          req_cols: list, opt_cols: list) -> str:
    req_str = "\n".join(f"    {r} = find_col_by_role(df, \"{r}\")" for r in req_cols)
    opt_str = "\n".join(f"    {r} = find_col_by_role(df, \"{r}\")" for r in opt_cols)
    if req_cols:
        pairs = ", ".join(f'("{r}", {r})' for r in req_cols)
        missing_check = (
            f"    missing = [r for r, c in [{pairs}] if c is None]\n"
            "    if missing:\n"
            '        return {"erreur": f"Colonnes requises absentes : {missing}"}\n'
        )
    else:
        missing_check = ""

    return (
        f'"""\ntools/.../{fn_name}.py\n{description}\n\n'
        'Interface : run(df, params) -> dict\n"""\n'
        'from __future__ import annotations\n\n'
        'import pandas as pd\n'
        'from agents.mortality.dictionary.column_schema import find_col_by_role\n\n\n'
        'def run(df: pd.DataFrame, params: dict | None = None) -> dict:\n'
        '    params = params or {}\n\n'
        + (req_str + "\n" if req_str else "")
        + (opt_str + "\n" if opt_str else "")
        + ("\n" + missing_check if missing_check else "")
        + "\n    # TODO : implémenter la logique\n    result = {}\n\n    return result\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Désambiguation (modal)
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("store-disambiguation", "data", allow_duplicate=True),
    Input("btn-validate-mapping", "n_clicks"),
    State("store-df-json", "data"),
    prevent_initial_call=True,
)
def open_mapping_validation(n_clicks, df_json):
    """Clic « Valider le mapping » → remplit store-disambiguation, ce qui
    ouvre le modal de mapping éditable (pré-rempli avec l'auto-détection).
    Le flag `source=validate_button` indique à submit_disambiguation de créer
    le clone (et non de relancer l'agent)."""
    if not n_clicks or not df_json:
        raise PreventUpdate
    from io import StringIO
    df = pd.read_json(StringIO(df_json), orient="split")
    report = build_mapping_report(df, get_capabilities())
    return {
        "source":                    "validate_button",
        "task_type":                 "mortality_table",
        "needs_column_mapping":      True,
        "needs_value_mapping":       False,
        "needs_form":                False,
        "df_columns":                list(df.columns),
        "column_mapping_suggestion": report["matched"],
        "form_fields":               [],
    }


@app.callback(
    Output("modal-disambiguation", "is_open"),
    Output("modal-disambiguation-body", "children"),
    Input("store-disambiguation", "data"),
    Input("btn-open-disambiguation", "n_clicks"),
    Input("btn-disambiguation-cancel", "n_clicks"),
    State("modal-disambiguation", "is_open"),
    prevent_initial_call=True,
)
def toggle_disambiguation_modal(disam_data, open_clicks, cancel_clicks, is_open):
    """Ouvre / ferme le modal et en rend le contenu."""
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "btn-disambiguation-cancel":
        return False, dash.no_update

    if trigger_id in ("store-disambiguation", "btn-open-disambiguation"):
        # Chercher les données dans le store si ouvert par bouton
        if not disam_data:
            raise PreventUpdate
        body_parts = []

        # ── Section mapping colonnes ─────────────────────────────────────────
        if disam_data.get("needs_column_mapping"):
            df_columns = disam_data.get("df_columns") or []
            suggestion = disam_data.get("column_mapping_suggestion") or {}
            if df_columns:
                body_parts.append(html.Div([
                    html.H6([html.I(className="fa fa-table me-2"), "Correspondance des colonnes CSV"],
                            className="text-secondary mb-3"),
                    _render_column_mapping_table(df_columns, suggestion),
                ]))

        # ── Section formulaire prérequis ─────────────────────────────────────
        if disam_data.get("needs_form"):
            form_fields = disam_data.get("form_fields") or []
            if form_fields:
                body_parts.append(_render_prerequisites_form(form_fields))

        if not body_parts:
            raise PreventUpdate

        return True, html.Div(body_parts)

    raise PreventUpdate


@app.callback(
    Output("modal-disambiguation", "is_open", allow_duplicate=True),
    Output("store-chat-history", "data", allow_duplicate=True),
    Output("interval-poll", "disabled", allow_duplicate=True),
    Output("agent-status-badge", "children", allow_duplicate=True),
    Output("agent-status-badge", "color", allow_duplicate=True),
    Output("store-last-event-idx", "data", allow_duplicate=True),
    Input("btn-disambiguation-confirm", "n_clicks"),
    State("store-disambiguation", "data"),
    State({"type": "col-mapping-select", "field": ALL}, "value"),
    State({"type": "col-mapping-select", "field": ALL}, "id"),
    State({"type": "prereq-input", "key": ALL}, "value"),
    State({"type": "prereq-input", "key": ALL}, "id"),
    State("store-chat-history", "data"),
    State("switch-step-mode", "value"),
    State("store-df-json", "data"),
    prevent_initial_call=True,
)
def submit_disambiguation(
    n_clicks, disam_data,
    col_values, col_ids,
    prereq_values, prereq_ids,
    history, step_mode, df_json,
):
    """
    Sauvegarde le mapping confirmé + les prérequis dans data_store,
    puis relance le thread agent pour continuer.
    """
    if not n_clicks:
        raise PreventUpdate

    # ── 1. Construire le column_mapping confirmé ─────────────────────────────
    col_mapping: dict[str, str] = {}
    for id_dict, value in zip(col_ids or [], col_values or []):
        field = id_dict.get("field", "")
        if field and value:
            col_mapping[field] = value

    # ── 1bis. Branche « Valider le mapping » : créer le clone normalisé ──────
    # Déclenchée par le bouton btn-validate-mapping (source=validate_button).
    # On crée la base de travail (clone) et on NE relance PAS l'agent — c'est
    # un acte de préparation, le calcul sera demandé séparément par l'user.
    if (disam_data or {}).get("source") == "validate_button":
        from io import StringIO
        from tools.conversation.apply_normalization import run as _apply_norm
        from session.memory_manager import MemoryManager
        from agents.mortality.agents._utils import msgpack_safe

        history = list(history or [])
        try:
            if not df_json:
                raise ValueError("Aucun fichier chargé.")
            df = pd.read_json(StringIO(df_json), orient="split")
            if len(df) == 0:
                raise ValueError("Fichier vide.")
            with _writer_lock:
                ds = _writer_state["data_store"]
                session_id = _writer_state.get("session_id")
                if col_mapping:
                    ds["column_mapping"] = col_mapping
                ds["column_mapping_confirmed"] = True
                ds["_dataset_ref"]             = session_id
                res = _apply_norm(df, {"force": True}, ds)
                if res.get("erreur"):
                    raise RuntimeError(res["erreur"])
                norm_ok   = bool(res.get("records_normalized"))
                norm_path = res.get("dataset_ref_normalized")
                audit     = (ds.get("_audit") or {}).get("normalization") or {}
                ds["mapping_validated"]      = True
                ds["records_normalized"]     = norm_ok
                ds["dataset_ref_normalized"] = norm_path
                # Crucial : `_apply_norm` peut écrire un value_mapping à clés
                # numpy.int (ex. sexe {np.int64(1): "H"}), ce que orjson refuse
                # à la sérialisation LangGraph (« Dict key must a type
                # serializable with OPT_NON_STR_KEYS »). On normalise ici, au
                # point d'entrée UI, parce que cette branche n'est pas un nœud
                # LangGraph et ne passe donc pas par sanitize_node_result.
                _writer_state["data_store"] = msgpack_safe(ds)
                ds = _writer_state["data_store"]
            # Persister dans le SessionState (durable au-delà de la session live)
            mm = MemoryManager(session_id)
            mm.load()
            mm.state.column_mapping           = col_mapping or mm.state.column_mapping
            mm.state.column_mapping_confirmed = True
            mm.state.records_normalized       = norm_ok
            mm.state.dataset_ref_normalized   = norm_path
            mm.state.cinematic_state["mapping_validated"] = True
            mm.save()
            history.append({"role": "assistant",
                             "content": _format_clone_message(audit)})
            return False, history, True, "Base de travail prête", "success", dash.no_update
        except Exception as exc:
            history.append({
                "role": "assistant",
                "content": f"⚠️ Échec de la création de la base de travail : {exc}",
            })
            return False, history, True, "Erreur", "danger", dash.no_update

    # ── 2. Construire les prérequis du formulaire ────────────────────────────
    prereqs: dict[str, str] = {}
    for id_dict, value in zip(prereq_ids or [], prereq_values or []):
        key = id_dict.get("key", "")
        if key and value is not None and str(value).strip():
            prereqs[key] = str(value).strip()

    # ── 3. Mettre à jour le data_store ──────────────────────────────────────
    task_type = (disam_data or {}).get("task_type", "mortality_table")
    target_agent = "writer" if task_type == "report" else "builder"

    with _writer_lock:
        ds = _writer_state["data_store"]
        # Mettre à jour le column_mapping uniquement si le formulaire a affiché
        # les dropdowns de mapping (needs_column_mapping=True). Sinon, conserver
        # le mapping confirmé lors de l'upload pour éviter une boucle.
        needs_col_mapping = (disam_data or {}).get("needs_column_mapping", False)
        if needs_col_mapping and col_mapping:
            ds["column_mapping"]           = col_mapping
            ds["column_mapping_confirmed"] = True
            for canonical, csv_col in col_mapping.items():
                ds[f"col_{canonical}"] = csv_col
        elif not ds.get("column_mapping_confirmed"):
            # Aucun mapping UI, mais pas encore confirmé — ne pas bloquer
            ds["column_mapping_confirmed"] = True

        # NE PAS poser _disambiguation_done ici : le master_node s'en charge
        # après avoir exécuté maybe_normalize_records. Poser ce flag
        # prématurément fait sauter la normalisation → input_records jamais
        # stocké → branche déterministe du Builder (US-20) skippée.
        # Injecter l'agent cible directement — _run_writer_in_thread ne l'écrasera pas
        ds["_initial_active_agent"]    = target_agent

        # Injecter dans study_plan
        sp = ds.setdefault("study_plan", {})
        for key, value in prereqs.items():
            sp[key] = value

        _writer_state["events"]  = []
        _writer_state["running"] = True
        _writer_state["step_by_step"] = bool(step_mode)

    # ── 4. Message de confirmation dans l'historique ─────────────────────────
    history = list(history or [])
    summary_parts = []
    if col_mapping:
        summary_parts.append(
            "Mapping colonnes confirmé : " +
            ", ".join(f"{k}={v}" for k, v in col_mapping.items() if v)
        )
    if prereqs:
        summary_parts.append(
            "Paramètres : " + ", ".join(f"{k}={v}" for k, v in prereqs.items())
        )
    summary = " | ".join(summary_parts)
    history.append({"role": "user", "content": f"[Formulaire confirmé] {summary}"})

    # ── 5. Relancer le thread agent ──────────────────────────────────────────
    t = threading.Thread(
        target=_run_writer_in_thread,
        args=(history, df_json),
        daemon=True,
    )
    t.start()

    return False, history, False, "En cours…", "warning", 0


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks — Modal Data Catalogue
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("modal-datacatalogue", "is_open"),
    Output("btn-open-datacatalogue", "style"),
    Output("dc-start-year", "value"),
    Output("dc-end-year",   "value"),
    Input("btn-open-datacatalogue", "n_clicks"),
    Input("btn-datacatalogue-cancel", "n_clicks"),
    Input("store-datacatalogue-trigger", "data"),
    State("modal-datacatalogue", "is_open"),
    State("dc-start-year", "value"),
    State("dc-end-year",   "value"),
    prevent_initial_call=True,
)
def toggle_datacatalogue_modal(open_clicks, cancel_clicks, trigger_data,
                                is_open, cur_start, cur_end):
    """Ouvre / ferme le modal data catalogue. Trois déclencheurs :

    - clic sur le bouton sidebar → ouvre simplement (sans modifier années).
    - clic Annuler → ferme.
    - event `datacatalogue_incomplete` reçu par poll_agent (re-routage du
      Builder bloqué) → ouvre AUTOMATIQUEMENT + pré-remplit les années à
      partir des suggestions (min/max année de décès auto-détectées à
      l'upload CSV) + RÉVÈLE le bouton sidebar (caché jusque-là).
    """
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    visible_style = {"display": "block"}  # bouton révélé

    # Garde-fou : le bouton est rendu dynamiquement par upload_csv. À sa
    # première apparition Dash peut déclencher cette callback avec
    # n_clicks=0/None malgré prevent_initial_call=True → modal s'ouvrirait
    # spontanément au chargement. On exige strictement n_clicks > 0.
    if trigger == "btn-open-datacatalogue":
        if not open_clicks:
            raise PreventUpdate
        return True, dash.no_update, dash.no_update, dash.no_update
    if trigger == "btn-datacatalogue-cancel":
        if not cancel_clicks:
            raise PreventUpdate
        return False, dash.no_update, dash.no_update, dash.no_update
    if trigger == "store-datacatalogue-trigger":
        # Avec la bulle inline (2026-05-28), on ne déclenche PLUS l'ouverture
        # auto du modal — la bulle gère le flux principal. Le store sert
        # uniquement à RÉVÉLER le bouton sidebar pour accès manuel rapide.
        if not (trigger_data or {}).get("reveal"):
            raise PreventUpdate
        return False, visible_style, dash.no_update, dash.no_update
    return is_open, dash.no_update, dash.no_update, dash.no_update


@app.callback(
    Output("dc-methods-explicit-container", "children"),
    Output("dc-methods-explicit-container", "style"),
    Input("dc-methods-mode", "value"),
    Input("dc-report-mode", "value"),
    Input("dc-gender", "value"),
    prevent_initial_call=False,
)
def render_dc_methods_explicit(methods_mode, report_mode, gender):
    """Affiche un Select par tool quand l'user choisit 'préciser'. Les
    tools listés sont dérivés du catalogue via method_choices_for_mode."""
    if methods_mode != "explicit":
        return [], {"display": "none"}
    try:
        from agents.master.method_choices import all_choices_for_mode
        choices = all_choices_for_mode(report_mode, gender)
    except Exception:
        choices = []
    if not choices:
        return [html.Div("Aucune méthode à choisir pour ce mode.",
                         className="text-muted small fst-italic")], \
               {"display": "block"}
    selects = []
    for c in choices:
        selects.append(html.Div([
            dbc.Label(f"{c.label} :", className="small fw-bold mt-2"),
            dbc.Select(
                id={"type": "dc-method-select", "tool": c.tool},
                options=[{"label": v, "value": v} for v in c.choices],
                value=c.default,
                size="sm",
            ),
        ]))
    return selects, {"display": "block"}


@app.callback(
    Output("modal-datacatalogue", "is_open", allow_duplicate=True),
    Output("store-chat-history", "data", allow_duplicate=True),
    Input("btn-datacatalogue-confirm", "n_clicks"),
    State("dc-report-mode",   "value"),
    State("dc-gender",        "value"),
    State("dc-write",         "value"),
    State("dc-methods-mode",  "value"),
    State({"type": "dc-method-select", "tool": ALL}, "value"),
    State({"type": "dc-method-select", "tool": ALL}, "id"),
    State("dc-start-year",    "value"),
    State("dc-end-year",      "value"),
    State("store-chat-history", "data"),
    prevent_initial_call=True,
)
def submit_datacatalogue(n_clicks, report_mode, gender, write,
                          methods_mode, method_values, method_ids,
                          start_year, end_year, history):
    """Stocke tous les choix utilisateur dans data_store en un coup.
    Le Builder pourra alors passer la gate au prochain appel."""
    if not n_clicks:
        raise PreventUpdate

    # Construire le dict des méthodes choisies (si mode explicit)
    methods_dict = {}
    for id_dict, value in zip(method_ids or [], method_values or []):
        tool = id_dict.get("tool", "")
        if tool and value:
            methods_dict[tool] = value

    history = list(history or [])
    summary_parts = []
    try:
        with _writer_lock:
            ds = _writer_state["data_store"]
            ds["report_mode"] = report_mode
            ds["_write"]      = write
            sp = ds.setdefault("study_plan", {})
            sp["gender_segmentation"] = gender
            sp["report_mode"]         = report_mode
            sp["write"]               = write
            if methods_mode == "auto":
                sp["methods_auto"] = True
                sp.pop("methods", None)
            else:
                sp["methods_auto"] = False
                sp["methods"] = methods_dict
            # Période d'observation (master_from_data avec confirm_with_user)
            if start_year is not None:
                sp["start_year"]                = int(start_year)
            if end_year is not None:
                sp["end_year"]                  = int(end_year)
            if start_year is not None and end_year is not None:
                sp["observation_period_years"]  = [int(start_year), int(end_year)]
                sp["num_observation_years"]     = int(end_year) - int(start_year) + 1
        summary_parts.append(f"mode={report_mode}, sexe={gender}, PDF={write}")
        if methods_mode == "auto":
            summary_parts.append("méthodes=auto")
        else:
            summary_parts.append(f"méthodes={len(methods_dict)} précisées")
        if start_year and end_year:
            summary_parts.append(f"période {start_year}–{end_year}")
    except Exception as exc:
        history.append({
            "role":    "assistant",
            "content": f"⚠️ Échec enregistrement data catalogue : {exc}",
        })
        return False, history

    history.append({
        "role":    "assistant",
        "content": ("✅ Data catalogue complété (" + " ; ".join(summary_parts)
                    + "). Vous pouvez maintenant lancer un calcul "
                    "(« construit le rapport », « calcule la table », …)."),
    })
    return False, history


# ─────────────────────────────────────────────────────────────────────────────
# Callback — Bulle inline du data catalogue (alternative au modal)
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("store-chat-history", "data", allow_duplicate=True),
    Input({"type": "dcf-confirm", "form_id": ALL}, "n_clicks"),
    State({"type": "dcf-confirm", "form_id": ALL}, "id"),
    State({"type": "dcf",         "form_id": ALL, "f": ALL}, "value"),
    State({"type": "dcf",         "form_id": ALL, "f": ALL}, "id"),
    State({"type": "dcf-method",  "form_id": ALL, "tool": ALL}, "value"),
    State({"type": "dcf-method",  "form_id": ALL, "tool": ALL}, "id"),
    State("store-chat-history", "data"),
    prevent_initial_call=True,
)
def submit_datacatalogue_bubble(confirm_clicks, confirm_ids,
                                  field_values, field_ids,
                                  method_values, method_ids, history):
    """Soumission inline du data catalogue depuis la bulle chat.

    Identifie quelle bulle a été soumise via callback_context.triggered,
    filtre les champs du même form_id, met à jour le data_store puis
    figés la bulle en mode confirmation."""
    ctx = callback_context
    if not ctx.triggered or not any(confirm_clicks or []):
        raise PreventUpdate

    # Identifier le form_id du bouton cliqué via prop_id
    import json as _json
    trig_prop = ctx.triggered[0]["prop_id"]
    if not trig_prop or trig_prop == ".":
        raise PreventUpdate
    id_part = trig_prop.rsplit(".", 1)[0]
    try:
        clicked_id = _json.loads(id_part)
    except Exception:
        raise PreventUpdate
    form_id = clicked_id.get("form_id")
    if not form_id:
        raise PreventUpdate

    # Récolter les valeurs des champs de CE form_id uniquement
    by_field: dict[str, object] = {}
    for fid, fval in zip(field_ids or [], field_values or []):
        if fid.get("form_id") != form_id:
            continue
        by_field[fid.get("f", "")] = fval

    report_mode = by_field.get("report-mode")   or "full_report"
    gender      = by_field.get("gender")        or "unisex"
    write       = by_field.get("write")         or "yes"
    methods_mode= by_field.get("methods-mode")  or "auto"
    start_year  = by_field.get("start-year")
    end_year    = by_field.get("end-year")

    history = list(history or [])

    # Construire le résumé pour la bulle confirmée
    summary_parts = [f"mode={report_mode}", f"sexe={gender}", f"PDF={write}",
                     f"méthodes={methods_mode}"]
    if start_year and end_year:
        summary_parts.append(f"période {start_year}–{end_year}")
    summary = " ; ".join(summary_parts)

    try:
        with _writer_lock:
            ds = _writer_state["data_store"]
            ds["report_mode"] = report_mode
            ds["_write"]      = write
            sp = ds.setdefault("study_plan", {})
            sp["gender_segmentation"] = gender
            sp["report_mode"]         = report_mode
            sp["write"]               = write
            if methods_mode == "auto":
                sp["methods_auto"] = True
                sp.pop("methods", None)
            else:
                # Mode explicit : récolter les selects par tool de CE form_id
                sp["methods_auto"] = False
                picked: dict[str, str] = {}
                for mid, mval in zip(method_ids or [], method_values or []):
                    if mid.get("form_id") != form_id:
                        continue
                    if mval:
                        picked[mid.get("tool", "")] = str(mval)
                if picked:
                    sp["methods"] = picked
            if start_year is not None and start_year != "":
                sp["start_year"] = int(start_year)
            if end_year is not None and end_year != "":
                sp["end_year"]   = int(end_year)
            if (start_year is not None and start_year != ""
                    and end_year is not None and end_year != ""):
                sp["observation_period_years"] = [int(start_year), int(end_year)]
                sp["num_observation_years"]    = int(end_year) - int(start_year) + 1
    except Exception as exc:
        # En cas d'erreur on garde le formulaire ouvert + message d'erreur
        history.append({"role": "assistant",
                        "content": f"⚠️ Échec enregistrement : {exc}"})
        return history

    # Figés la bulle correspondante en mode confirmation
    for entry in history:
        if (entry.get("role") == "_datacatalogue_form"
                and entry.get("form_id") == form_id):
            entry["submitted"]         = True
            entry["submitted_summary"] = summary
            break

    # Message de suite pour inviter à relancer le calcul
    history.append({
        "role":    "assistant",
        "content": ("✅ Data catalogue complété. Tu peux maintenant relancer "
                    "ta demande de calcul — j'ai tout ce qu'il me faut."),
    })
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Callback — Bulle décision tool (smoothing : violations monotonie, etc.)
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("store-chat-history", "data", allow_duplicate=True),
    Output("interval-poll", "disabled", allow_duplicate=True),
    Output("agent-status-badge", "children", allow_duplicate=True),
    Output("agent-status-badge", "color", allow_duplicate=True),
    Output("store-last-event-idx", "data", allow_duplicate=True),
    Input({"type": "dr-submit", "form_id": ALL}, "n_clicks"),
    State({"type": "dr-option", "form_id": ALL}, "value"),
    State({"type": "dr-option", "form_id": ALL}, "id"),
    State("store-chat-history", "data"),
    State("store-df-json", "data"),
    prevent_initial_call=True,
)
def submit_decision_required(submit_clicks, option_values, option_ids,
                              history, df_json):
    """Soumission d'une bulle décision (decision_required du Builder).

    Met à jour study_plan selon l'option choisie, pop le verrou
    `_pending_decision` du data_store et injecte un message user pour
    relancer le pipeline. Plan refonte garde-fou 2026-06-03.
    """
    ctx = callback_context
    if not ctx.triggered or not any(submit_clicks or []):
        raise PreventUpdate

    import json as _json
    trig_prop = ctx.triggered[0]["prop_id"]
    if not trig_prop or trig_prop == ".":
        raise PreventUpdate
    id_part = trig_prop.rsplit(".", 1)[0]
    try:
        clicked_id = _json.loads(id_part)
    except Exception:
        raise PreventUpdate
    form_id = clicked_id.get("form_id")
    if not form_id:
        raise PreventUpdate

    chosen = None
    for oid, oval in zip(option_ids or [], option_values or []):
        if oid.get("form_id") == form_id and oval:
            chosen = oval
            break
    if not chosen:
        raise PreventUpdate

    history = list(history or [])
    try:
        with _writer_lock:
            ds = _writer_state["data_store"]
            # Pop le verrou — graphe accepte à nouveau de re-router.
            ds.pop("_pending_decision", None)
            sp = ds.setdefault("study_plan", {})

            if chosen == "increase_lambda":
                current = float(sp.get("smoothing_lambda") or 100)
                sp["smoothing_lambda"] = current * 2
            elif chosen == "change_method":
                # v1 : Gompertz par défaut. Sous-sélecteur fin à venir.
                sp.setdefault("methods", {})["builder.smoothing"] = "gompertz"
            elif chosen == "accept_with_note":
                sp["smoothing_accept_violations"] = True
    except Exception as exc:
        history.append({"role": "assistant",
                        "content": f"⚠️ Échec enregistrement décision : {exc}"})
        return history

    # Figer la bulle correspondante
    for entry in history:
        if (entry.get("role") == "_decision_required_panel"
                and entry.get("form_id") == form_id):
            entry["submitted"] = True
            entry["chosen"]    = chosen
            break

    # Message user injecté = relance naturelle du graphe LangGraph
    label_map = {
        "increase_lambda":  "Augmenter le paramètre de lissage",
        "change_method":    "Changer de méthode de lissage (Gompertz)",
        "accept_with_note": "Accepter la table en l'état (mention dans le rapport)",
    }
    history.append({
        "role":    "user",
        "content": f"✅ J'ai validé : {label_map.get(chosen, chosen)}. "
                   "Poursuis le calcul.",
    })

    # Relance automatique du graphe LangGraph (équivalent send_message).
    # Sans ça, l'user devrait re-taper un message — on évite le frottement.
    with _writer_lock:
        _writer_state["events"]  = []
        _writer_state["running"] = True
        _writer_state["pending_tool_call"] = None
    threading.Thread(
        target=_run_writer_in_thread,
        args=(history, df_json),
        daemon=True,
    ).start()
    return history, False, "En cours…", "warning", 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _warmup_doctrine_retriever() -> None:
    """Pré-charge le modèle embedder + index FAISS pour la doctrine
    actuarielle. Sans ça le 1er appel utilisateur à
    conversation.search_doctrine subit un cold start de ~5s.
    Lancé en background pour ne pas bloquer le démarrage de l'app."""
    try:
        from tools.conversation.search_doctrine import warmup
        ok = warmup()
        print(f"[warmup] Retriever doctrine prêt = {ok}", flush=True)
    except Exception as exc:
        print(f"[warmup] doctrine échec : {exc}", flush=True)


if __name__ == "__main__":
    import os
    # HOTFIX-pre-refacto-2026-05 (Bug 4) : Flask debug mode fork un reloader
    # process. Sans condition correcte, warmup s'exécute 2× (parent + child).
    # Logique : si on est en debug Flask, lancer UNIQUEMENT dans le reloader
    # child (WERKZEUG_RUN_MAIN posé). Sinon (prod), lancer normalement.
    flask_debug = os.environ.get("FLASK_DEBUG") == "1"
    is_reloader_child = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if (not flask_debug) or is_reloader_child:
        threading.Thread(target=_warmup_doctrine_retriever, daemon=True).start()
    app.run(debug=True, host="0.0.0.0", port=8050)
