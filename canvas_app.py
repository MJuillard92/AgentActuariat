"""
canvas_app.py
Interface principale — 2 onglets :
  • Rapport guidé : dialogue avec le WriterAgent (upload CSV + chat)
  • DEV           : gestion des capacités actuarielles (cards + éditeur de code)
"""
from __future__ import annotations

import base64
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

from report_agent.tools.tool_registry import get_capabilities
from report_agent.dictionary.column_schema import COLUMN_SCHEMA, build_mapping_report

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

_writer_state: dict = {"events": [], "running": False}
_writer_lock = threading.Lock()


def _run_writer_in_thread(history: list[dict], df_json: str | None) -> None:
    from report_agent.writer_agent import WriterAgent
    df = None
    if df_json:
        try:
            df = pd.read_json(StringIO(df_json), orient="split")
        except Exception:
            pass

    writer = WriterAgent()
    try:
        for event in writer.run_agent_loop(history, df=df):
            with _writer_lock:
                _writer_state["events"].append(event)
    except Exception as exc:
        with _writer_lock:
            _writer_state["events"].append({"type": "error", "message": str(exc)})
    finally:
        with _writer_lock:
            _writer_state["running"] = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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
            # ── Panneau gauche : CSV ─────────────────────────────────────────
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
                                "padding": "20px",
                                "cursor": "pointer",
                                "backgroundColor": "#FAFAFA",
                            },
                            multiple=False,
                        ),
                        html.Div(id="csv-info", className="mt-3"),
                    ]),
                ], className="mb-3"),
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
                    dbc.CardFooter([
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
            ], width=9),
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
    tools_root = Path(__file__).parent / "report_agent" / "tools"
    dict_root = Path(__file__).parent / "report_agent" / "dictionary"
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

    # Polling interval (désactivé par défaut)
    dcc.Interval(id="interval-poll", interval=400, n_intervals=0, disabled=True),

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
    prevent_initial_call=True,
)
def send_message(n_clicks, message, history, df_json, _last_idx):
    if not n_clicks or not message or not message.strip():
        raise PreventUpdate

    history = history or []
    history.append({"role": "user", "content": message.strip()})

    # Réinitialiser le state
    with _writer_lock:
        _writer_state["events"] = []
        _writer_state["running"] = True

    # Lancer le thread
    t = threading.Thread(
        target=_run_writer_in_thread,
        args=(history, df_json),
        daemon=True,
    )
    t.start()

    return history, "", False, "En cours…", "warning", 0


@app.callback(
    Output("chat-messages", "children"),
    Output("interval-poll", "disabled", allow_duplicate=True),
    Output("agent-status-badge", "children", allow_duplicate=True),
    Output("agent-status-badge", "color", allow_duplicate=True),
    Output("store-chat-history", "data", allow_duplicate=True),
    Output("store-last-event-idx", "data", allow_duplicate=True),
    Input("interval-poll", "n_intervals"),
    State("store-chat-history", "data"),
    State("store-last-event-idx", "data"),
    prevent_initial_call=True,
)
def poll_agent(n_intervals, history, last_idx):
    with _writer_lock:
        events = list(_writer_state["events"])
        running = _writer_state["running"]

    history = list(history or [])
    new_events = events[last_idx:]

    for ev in new_events:
        ev_type = ev.get("type")
        if ev_type == "message":
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
            image_b64 = result.get("image_b64")
            result_keys = [k for k in result if k not in ("erreur", "image_b64")]
            history.append({
                "role": "_tool_result",
                "function_name": ev.get("function_name", ""),
                "image_b64": image_b64,
                "result_keys": result_keys,
                "content": "",
            })
        elif ev_type == "error":
            history.append({"role": "assistant", "content": f"⚠️ Erreur : {ev.get('message', '')}"})

    new_idx = len(events)

    # Construire les bulles
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
            img = h.get("image_b64")
            if img:
                bubbles.append(_chat_bubble("assistant", "", extra={
                    "type": "tool_result",
                    "image_b64": img,
                    "function_name": h.get("function_name", ""),
                }))
            elif h.get("result_keys"):
                bubbles.append(_chat_bubble("assistant", "", extra={
                    "type": "tool_result",
                    "function_name": h.get("function_name", ""),
                    "result_keys": h.get("result_keys", []),
                }))

    done = not running
    poll_disabled = done
    status_text = "Prêt" if done else "En cours…"
    status_color = "success" if done else "warning"

    return bubbles, poll_disabled, status_text, status_color, history, new_idx


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
    Output("dev-file-path-display", "value"),
    Output("dev-code-editor", "value"),
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
    tools_root = Path(__file__).parent / "report_agent" / "tools"
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

        tools_root = Path(__file__).parent / "report_agent" / "tools"
        target_dir = tools_root / tool_name
        target_dir.mkdir(parents=True, exist_ok=True)
        py_path = target_dir / f"{fn_name}.py"
        py_path.write_text(code, encoding="utf-8")

        # Mettre à jour builder_capabilities.json
        caps_path = Path(__file__).parent / "report_agent" / "builder_capabilities.json"
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
        f'"""\nreport_agent/tools/.../{fn_name}.py\n{description}\n\n'
        'Interface : run(df, params) -> dict\n"""\n'
        'from __future__ import annotations\n\n'
        'import pandas as pd\n'
        'from report_agent.dictionary.column_schema import find_col_by_role\n\n\n'
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
