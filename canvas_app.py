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

_SESSIONS_DIR = Path(__file__).parent / "sessions"

_writer_state: dict = {
    "events": [], "running": False, "data_store": {}, "context_docs": [],
    "step_by_step": False, "pending_tool_call": None,
    "session_id": None,    # yymmddhhmm — set on CSV upload or first tool call
    "csv_filename": None,
}
_writer_lock = threading.Lock()


def _new_session_id() -> str:
    return datetime.datetime.now().strftime("%y%m%d%H%M")


def _save_session(session_id: str, data_store: dict, csv_filename: str | None) -> None:
    """Persiste le data_store sur disque après chaque tool call."""
    if not session_id or not data_store:
        return
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id":   session_id,
        "timestamp":    datetime.datetime.now().isoformat(),
        "csv_filename": csv_filename,
        "n_tool_calls": len(data_store.get("_call_log", [])),
        "data_store":   data_store,
    }
    path = _SESSIONS_DIR / f"{session_id}.json"
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, default=str, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def load_session(session_id: str) -> dict | None:
    """
    Charge un data_store depuis une session persistée.

    Usage (REPL ou script) :
        from canvas_app import load_session
        ds = load_session("2604021530")
        print(ds["data_store"].keys())
    """
    path = _SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_session_context(session_id: str, payload: dict, data_store: dict) -> str:
    """Génère un message de contexte synthétique lisible par l'agent après restauration."""
    import math

    csv_filename = payload.get("csv_filename", "—")
    n_calls      = payload.get("n_tool_calls", 0)

    key_labels = {
        "exposure_table": "Table d'exposition (builder.exposure)",
        "qx_table":       "Taux bruts (builder.crude_rates)",
        "smoothed_table": "Table lissée (builder.smoothing)",
        "diagnostics":    "Diagnostics de crédibilité (builder.diagnostics)",
        "validation":     "Validation statistique (builder.validation)",
        "benchmarking":   "Benchmarking (builder.benchmarking)",
        "certification_report": "Rapport PDF généré (build_pdf.certification_report)",
    }
    computed = [label for key, label in key_labels.items() if data_store.get(key)]

    lines = [
        f"[Session restaurée : {session_id}]",
        f"Fichier analysé : {csv_filename}",
        f"Nombre d'appels de tools : {n_calls}",
        "",
        "Calculs disponibles dans le data_store :",
    ]
    for label in computed:
        lines.append(f"  ✓ {label}")

    # Détails clés
    exposure_table = data_store.get("exposure_table") or []
    if exposure_table:
        ages_with_exp = [r.get("age") for r in exposure_table
                         if isinstance(r, dict) and (r.get("E_x") or 0) > 0]
        if ages_with_exp:
            e_total = sum((r.get("E_x") or 0) for r in exposure_table)
            d_total = sum((r.get("D_x") or 0) for r in exposure_table)
            lines.append(
                f"\nExposition : {min(ages_with_exp)}-{max(ages_with_exp)} ans, "
                f"{len(ages_with_exp)} âges, {e_total:,.0f} P-A, {int(d_total)} décès"
            )

    # Méthode de lissage
    for key in ("smoothing_method", "method"):
        method = data_store.get(key)
        if method:
            lam = data_store.get("lambda_wh") or data_store.get("lambda")
            n_nm = data_store.get("n_non_monotone", 0) or 0
            lines.append(
                f"Lissage : {method}"
                + (f", λ={lam}" if lam else "")
                + f", violations monotonie : {n_nm}"
            )
            break

    # SMR global
    benchmarking = data_store.get("benchmarking") or {}
    if isinstance(benchmarking, dict):
        smr = benchmarking.get("smr_global")
        if smr is not None and not (isinstance(smr, float) and math.isnan(smr)):
            pct = abs(1.0 - smr) * 100
            direction = "sous-mortalité" if smr < 1.0 else "sur-mortalité"
            ref = benchmarking.get("reference_name", "TH0002")
            lines.append(f"SMR global : {smr:.3f} ({direction} de {pct:.1f}% vs {ref})")

    # Dernier message de raisonnement (résumé)
    reasoning_log = data_store.get("_reasoning_log") or []
    if reasoning_log:
        last = reasoning_log[-1]
        snippet = last[:400].rstrip() + ("…" if len(last) > 400 else "")
        lines += ["", "Dernier état de la session :", snippet]

    lines += [
        "",
        "Vous pouvez maintenant poser des questions sur ces résultats ou "
        "demander la génération du rapport de certification.",
    ]
    return "\n".join(lines)


def restore_session(session_id: str) -> tuple[str, list[dict]]:
    """
    Réinjecte le data_store d'une session passée dans _writer_state.
    Retourne (message_statut, historique_chat_initial).

    L'historique initial contient un message de contexte synthétique pour que
    l'agent sache ce qui a été calculé dans cette session.
    """
    payload = load_session(session_id)
    if payload is None:
        return f"Session introuvable : {session_id}", []

    data_store = payload.get("data_store", {})
    if not data_store:
        return f"Session {session_id} vide ou corrompue.", []

    # Nettoyer les valeurs NaN sérialisées en JSON (non standard)
    import math
    def _clean(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj
    data_store = _clean(data_store)

    with _writer_lock:
        _writer_state["data_store"]   = data_store
        _writer_state["session_id"]   = session_id
        _writer_state["csv_filename"] = payload.get("csv_filename")

    keys = [k for k in data_store if not k.startswith("_")]
    n_calls = len(data_store.get("_call_log", []))
    status = (
        f"Session {session_id} restaurée — "
        f"{n_calls} appels, clés : {', '.join(keys)}"
    )

    context_msg = _build_session_context(session_id, payload, data_store)
    restored_history = [{"role": "assistant", "content": context_msg}]
    return status, restored_history


def list_sessions() -> list[dict]:
    """Retourne la liste des sessions disponibles, triée par date décroissante."""
    if not _SESSIONS_DIR.exists():
        return []
    sessions = []
    for p in sorted(_SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({
                "session_id":   raw.get("session_id", p.stem),
                "timestamp":    raw.get("timestamp", "")[:16].replace("T", " "),
                "csv_filename": raw.get("csv_filename", "—"),
                "n_tool_calls": raw.get("n_tool_calls", 0),
            })
        except Exception:
            continue
    return sessions

# Synchronisation mode pas à pas
_step_approval_event: threading.Event = threading.Event()
_step_cancel_flag: list[bool] = [False]


def _run_writer_in_thread(history: list[dict], df_json: str | None) -> None:
    from agents.mortality.agents.graph import stream_agent
    df = None
    if df_json:
        try:
            df = pd.read_json(StringIO(df_json), orient="split")
        except Exception:
            pass

    # Récupérer le data_store et context_docs persistés de la session
    with _writer_lock:
        data_store   = _writer_state["data_store"]
        context_docs = _writer_state["context_docs"]
        step_by_step = _writer_state["step_by_step"]
        # Générer un session_id si ce n'est pas encore fait (premier run sans upload CSV)
        if not _writer_state["session_id"]:
            _writer_state["session_id"] = _new_session_id()
        session_id   = _writer_state["session_id"]
        csv_filename = _writer_state["csv_filename"]

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

    try:
        for event in stream_agent(
            history, df=df, data_store=data_store, context_docs=context_docs,
            step_by_step=step_by_step,
            approval_event=_step_approval_event if step_by_step else None,
            cancel_flag=_step_cancel_flag if step_by_step else None,
        ):
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
                # Persister le data_store après chaque tool call complété
                if event["type"] == "tool_result":
                    _save_session(session_id, data_store, csv_filename)
    except Exception as exc:
        with _writer_lock:
            _writer_state["events"].append({"type": "error", "message": str(exc)})
    finally:
        with _writer_lock:
            _writer_state["running"] = False
            _writer_state["pending_tool_call"] = None


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
    """Construit les cards de capacités depuis builder_capabilities.json."""
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
# Layout
# ─────────────────────────────────────────────────────────────────────────────

app.layout = dbc.Container([
    # Stores
    dcc.Store(id="store-df-json"),
    dcc.Store(id="store-chat-history", data=[]),
    dcc.Store(id="store-last-event-idx", data=0),
    dcc.Store(id="store-pdf-path"),
    dcc.Store(id="store-txt-path"),
    dcc.Store(id="store-notebook-path"),
    dcc.Store(id="store-context-docs", data=[]),
    dcc.Store(id="store-step-mode", data=False),
    dcc.Store(id="store-agent-internals", data=[]),

    # Téléchargements
    dcc.Download(id="download-pdf"),
    dcc.Download(id="download-txt"),
    dcc.Download(id="download-notebook"),

    # Polling interval (désactivé par défaut)
    dcc.Interval(id="interval-poll", interval=400, n_intervals=0, disabled=True),

    # Interval one-shot pour attacher l'écouteur Enter sur chat-input
    dcc.Interval(id="init-listeners", interval=600, n_intervals=0, max_intervals=1, disabled=False),

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
    ], id="main-tabs", active_tab="tab-writer"),

], fluid=True, className="px-0")


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
    Output("store-df-json", "data"),
    Output("csv-info", "children"),
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

    # Réinitialiser le data_store et ouvrir une nouvelle session horodatée
    with _writer_lock:
        _writer_state["data_store"]   = {}
        _writer_state["session_id"]   = _new_session_id()
        _writer_state["csv_filename"] = filename

    caps = get_capabilities()
    report = build_mapping_report(df, caps)

    ready_fns = sum(1 for s in report["fn_readiness"].values() if s["ready"])
    total_fns = len(report["fn_readiness"])

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
    ])
    return df_json, info


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
    pdf_path = txt_path = notebook_path = None

    # Badge agent courant (dernier agent_switch vu)
    current_agent = None

    # Accumuler les nouvelles entrées internals
    new_internals = list(existing_internals or [])

    for ev in new_events:
        ev_type = ev.get("type")

        # Panneau internals : toujours loguer
        new_internals.append(_internals_entry(ev))

        if ev_type == "agent_switch":
            current_agent = ev.get("agent")

        elif ev_type == "message":
            history.append({"role": "assistant", "content": ev.get("content", "")})

        elif ev_type == "tool_call":
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

    done = not running
    poll_disabled = done

    # Badge principal : agent actif pendant le run, "Prêt" quand terminé
    if done:
        status_text  = "Prêt"
        status_color = "success"
    elif current_agent:
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
            banner, new_internals, internals_badge)


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

        # Mettre à jour builder_capabilities.json
        caps_path = Path(__file__).parent / "tools" / "builder_capabilities.json"
        caps = json.loads(caps_path.read_text(encoding="utf-8"))
        if tool_name not in caps["tools"]:
            caps["tools"][tool_name] = {"description": "", "functions": {}}

        try:
            parsed_params = json.loads(params_json) if params_json and params_json.strip() else {}
        except Exception:
            parsed_params = {}

        caps["tools"][tool_name]["functions"][fn_name] = {
            "description": fn_desc or "",
            "required_columns": req_cols or [],
            "optional_columns": opt_cols or [],
            "params": parsed_params,
        }
        caps_path.write_text(json.dumps(caps, ensure_ascii=False, indent=2), encoding="utf-8")

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
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
