"""
inspector_app.py
================
Application Dash standalone pour inspecter pas-à-pas les runs de l'agent.

Lancement :
    python inspector/inspector_app.py        # http://localhost:8051

Fonctionnalités :
  - Explorateur de variables (gauche) : toutes les tables/figures/scalaires du kernel
  - Visualiseur (centre) : DataTable, image, JSON selon le type
  - Log des étapes (bas) : code exécuté + sortie + statut pour chaque étape
  - Code cell : exécuter du code ad-hoc sur les données chargées
  - Export CSV de la variable sélectionnée
"""

from __future__ import annotations

import io
import json
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, ctx, dash_table, dcc, html
from dash.exceptions import PreventUpdate

# Permettre l'import depuis la racine du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from inspector.kernel_snapshot import (
    build_exec_namespace,
    list_sessions,
    load_manifest,
    load_steps,
    load_variable,
)

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Agent Inspector",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers UI
# ─────────────────────────────────────────────────────────────────────────────

def _var_badge(kind: str) -> dbc.Badge:
    colors = {
        "DataFrame": "primary",
        "ndarray":   "info",
        "Figure":    "success",
        "dict":      "warning",
        "scalar":    "secondary",
        "str":       "light",
        "list":      "dark",
    }
    return dbc.Badge(kind, color=colors.get(kind, "secondary"), className="ms-1")


def _build_var_list(session_id: str | None) -> html.Div:
    if not session_id:
        return html.Div("Sélectionne une session.", className="text-muted p-2")

    manifest = load_manifest(session_id)
    if not manifest:
        return html.Div("Aucune variable dans ce snapshot.", className="text-muted p-2")

    items = []
    for entry in manifest:
        name = entry["name"]
        kind = entry["type"]
        icon = entry.get("icon", "📦")

        # Sous-titre selon le type
        if kind == "DataFrame":
            shape = entry.get("shape", [])
            subtitle = f"{shape[0]:,} × {shape[1]}" if len(shape) == 2 else ""
        elif kind == "ndarray":
            subtitle = "×".join(str(s) for s in entry.get("shape", []))
        elif kind == "scalar":
            subtitle = str(entry.get("value", ""))
        else:
            subtitle = ""

        items.append(
            dbc.ListGroupItem(
                [
                    html.Span(f"{icon} {name}", className="fw-bold"),
                    _var_badge(kind),
                    html.Br(),
                    html.Small(subtitle, className="text-muted"),
                ],
                id={"type": "var-item", "name": name},
                action=True,
                className="py-2",
                style={"cursor": "pointer"},
            )
        )

    return dbc.ListGroup(items, flush=True)


def _build_step_accordion(session_id: str | None) -> dbc.Accordion:
    if not session_id:
        return dbc.Accordion([])

    steps = load_steps(session_id)
    items = []
    for i, step in enumerate(steps):
        success = step.get("success", True)
        icon = "✅" if success else "❌"
        desc = step.get("description", f"Étape {i+1}")
        code = step.get("code", "")
        output = step.get("output", "")
        ts = step.get("ts", "")

        items.append(
            dbc.AccordionItem(
                [
                    html.Small(ts, className="text-muted d-block mb-2"),
                    dbc.Tabs([
                        dbc.Tab([
                            html.Pre(
                                code,
                                style={
                                    "backgroundColor": "#f8f9fa",
                                    "padding": "10px",
                                    "fontSize": "12px",
                                    "maxHeight": "300px",
                                    "overflowY": "auto",
                                    "borderRadius": "4px",
                                },
                            )
                        ], label="Code"),
                        dbc.Tab([
                            html.Pre(
                                output,
                                style={
                                    "backgroundColor": "#f8f9fa",
                                    "padding": "10px",
                                    "fontSize": "12px",
                                    "maxHeight": "300px",
                                    "overflowY": "auto",
                                    "borderRadius": "4px",
                                    "color": "#dc3545" if not success else "#212529",
                                },
                            )
                        ], label="Sortie"),
                    ]),
                ],
                title=f"{icon} Étape {i+1} — {desc[:60]}",
            )
        )

    return dbc.Accordion(items, start_collapsed=True, always_open=False)


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

app.layout = dbc.Container(
    fluid=True,
    children=[
        # ── Header ───────────────────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(html.H4("🔍 Agent Inspector", className="mb-0"), width="auto"),
                dbc.Col(
                    dcc.Dropdown(
                        id="session-dropdown",
                        placeholder="Sélectionner une session…",
                        clearable=False,
                        style={"fontSize": "13px"},
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.Button("↻ Rafraîchir", id="refresh-btn", size="sm", color="secondary"),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Button("💾 Export CSV", id="export-btn", size="sm", color="primary", disabled=True),
                    width="auto",
                ),
                dcc.Download(id="download-csv"),
            ],
            align="center",
            className="py-2 border-bottom mb-2",
        ),

        # ── Corps principal ───────────────────────────────────────────────────
        dbc.Row(
            [
                # Gauche : explorateur de variables
                dbc.Col(
                    [
                        html.H6("Variables du kernel", className="text-uppercase text-muted fw-bold mb-2"),
                        html.Div(id="var-list", style={"overflowY": "auto", "maxHeight": "60vh"}),
                    ],
                    width=3,
                    className="border-end pe-2",
                ),

                # Droite : visualiseur
                dbc.Col(
                    [
                        html.Div(id="var-title", className="text-muted mb-2"),
                        html.Div(
                            id="var-viewer",
                            style={"overflowY": "auto", "maxHeight": "60vh"},
                            children=html.Div(
                                "← Clique sur une variable pour la visualiser.",
                                className="text-muted p-3",
                            ),
                        ),
                    ],
                    width=9,
                ),
            ],
            className="mb-3",
        ),

        html.Hr(),

        # ── Log des étapes ────────────────────────────────────────────────────
        dbc.Row(
            dbc.Col(
                [
                    html.H6("📋 Étapes de l'agent", className="text-uppercase text-muted fw-bold mb-2"),
                    html.Div(id="steps-accordion"),
                ]
            )
        ),

        html.Hr(),

        # ── Code cell ────────────────────────────────────────────────────────
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6("💻 Code cell", className="text-uppercase text-muted fw-bold mb-2"),
                            html.Small(
                                "Toutes les variables du snapshot sont disponibles directement (df_exposure, df_qx…)",
                                className="text-muted d-block mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Textarea(
                                            id="code-input",
                                            placeholder="df_exposure.describe()\n# ou\ndf_qx[df_qx['E_x'] > 100]",
                                            style={
                                                "width": "100%",
                                                "height": "100px",
                                                "fontFamily": "monospace",
                                                "fontSize": "13px",
                                            },
                                        ),
                                        width=10,
                                    ),
                                    dbc.Col(
                                        dbc.Button("▶ Run", id="run-btn", color="success", className="w-100"),
                                        width=2,
                                        className="d-flex align-items-center",
                                    ),
                                ]
                            ),
                            html.Div(id="code-output", className="mt-2"),
                        ]
                    )
                )
            )
        ),

        # ── Stores cachés ──────────────────────────────────────────────────
        dcc.Store(id="selected-var-name"),
        dcc.Interval(id="auto-refresh", interval=10_000, n_intervals=0),
    ],
    style={"paddingTop": "10px"},
)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("session-dropdown", "options"),
    Output("session-dropdown", "value"),
    Input("refresh-btn", "n_clicks"),
    Input("auto-refresh", "n_intervals"),
    State("session-dropdown", "value"),
)
def refresh_sessions(_clicks, _interval, current_value):
    sessions = list_sessions()
    options = [{"label": s, "value": s} for s in sessions]
    # Garder la session courante si elle existe toujours, sinon prendre la plus récente
    value = current_value if current_value in sessions else (sessions[0] if sessions else None)
    return options, value


@app.callback(
    Output("var-list", "children"),
    Output("steps-accordion", "children"),
    Input("session-dropdown", "value"),
    Input("auto-refresh", "n_intervals"),
)
def update_session_content(session_id, _interval):
    return _build_var_list(session_id), _build_step_accordion(session_id)


@app.callback(
    Output("selected-var-name", "data"),
    Input({"type": "var-item", "name": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_variable(n_clicks_list):
    if not any(n for n in n_clicks_list if n):
        raise PreventUpdate
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict):
        return triggered["name"]
    raise PreventUpdate


@app.callback(
    Output("var-viewer", "children"),
    Output("var-title", "children"),
    Output("export-btn", "disabled"),
    Input("selected-var-name", "data"),
    State("session-dropdown", "value"),
    prevent_initial_call=True,
)
def display_variable(var_name, session_id):
    if not var_name or not session_id:
        raise PreventUpdate

    manifest = load_manifest(session_id)
    entry = next((e for e in manifest if e["name"] == var_name), None)
    if entry is None:
        raise PreventUpdate

    kind = entry["type"]
    icon = entry.get("icon", "📦")
    title = f"{icon} {var_name}  —  {kind}"
    can_export = kind == "DataFrame"

    val = load_variable(session_id, var_name)
    if val is None:
        return html.Div("Impossible de charger la variable.", className="text-danger"), title, True

    # ── DataFrame ─────────────────────────────────────────────────────────
    if kind == "DataFrame":
        df = val
        title += f"  ({df.shape[0]:,} × {df.shape[1]})"
        table = dash_table.DataTable(
            data=df.head(500).round(6).to_dict("records"),
            columns=[{"name": c, "id": c} for c in df.columns],
            page_size=20,
            style_table={"overflowX": "auto"},
            style_cell={"fontSize": "12px", "fontFamily": "monospace", "padding": "4px 8px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5"},
            filter_action="native",
            sort_action="native",
        )
        stats = df.describe().round(4)
        stats_table = dash_table.DataTable(
            data=stats.reset_index().rename(columns={"index": "stat"}).to_dict("records"),
            columns=[{"name": c, "id": c} for c in ["stat"] + list(df.select_dtypes("number").columns)],
            style_cell={"fontSize": "11px", "fontFamily": "monospace", "padding": "3px 6px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5"},
        )
        viewer = html.Div([
            dbc.Tabs([
                dbc.Tab(table, label=f"Données (500 premières lignes)"),
                dbc.Tab(stats_table, label="Stats descriptives"),
            ])
        ])

    # ── Figure PNG ────────────────────────────────────────────────────────
    elif kind == "Figure":
        png_path = Path(val)
        if png_path.exists():
            import base64
            b64 = base64.b64encode(png_path.read_bytes()).decode()
            viewer = html.Img(
                src=f"data:image/png;base64,{b64}",
                style={"maxWidth": "100%"},
            )
        else:
            viewer = html.Div("Fichier PNG introuvable.", className="text-danger")

    # ── ndarray ───────────────────────────────────────────────────────────
    elif kind == "ndarray":
        arr = val
        title += f"  shape={arr.shape}  dtype={arr.dtype}"
        df_arr = pd.DataFrame(arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr)
        viewer = dash_table.DataTable(
            data=df_arr.head(200).round(6).to_dict("records"),
            columns=[{"name": str(c), "id": str(c)} for c in df_arr.columns],
            page_size=20,
            style_cell={"fontSize": "12px", "fontFamily": "monospace"},
        )

    # ── Dict / list / scalar / str ────────────────────────────────────────
    else:
        text = json.dumps(val, ensure_ascii=False, indent=2, default=str)
        viewer = html.Pre(
            text,
            style={
                "backgroundColor": "#f8f9fa",
                "padding": "12px",
                "borderRadius": "4px",
                "fontSize": "12px",
                "maxHeight": "55vh",
                "overflowY": "auto",
            },
        )

    return viewer, title, not can_export


@app.callback(
    Output("download-csv", "data"),
    Input("export-btn", "n_clicks"),
    State("selected-var-name", "data"),
    State("session-dropdown", "value"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, var_name, session_id):
    if not n_clicks or not var_name or not session_id:
        raise PreventUpdate
    val = load_variable(session_id, var_name)
    if not isinstance(val, pd.DataFrame):
        raise PreventUpdate
    return dcc.send_data_frame(val.to_csv, f"{var_name}.csv", index=False)


@app.callback(
    Output("code-output", "children"),
    Input("run-btn", "n_clicks"),
    State("code-input", "value"),
    State("session-dropdown", "value"),
    prevent_initial_call=True,
)
def run_code(n_clicks, code, session_id):
    if not n_clicks or not code or not session_id:
        raise PreventUpdate

    ns = build_exec_namespace(session_id)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(code, ns)  # noqa: S102
        output = buf.getvalue()

        # Si la dernière expression est un DataFrame → afficher comme table
        last_val = None
        try:
            last_line = [l.strip() for l in code.strip().splitlines() if l.strip()][-1]
            if not any(last_line.startswith(kw) for kw in ("import", "from", "#", "print")):
                last_val = eval(last_line, ns)  # noqa: S307
        except Exception:
            pass

        children = []
        if output:
            children.append(html.Pre(output, style={"fontSize": "12px", "backgroundColor": "#f8f9fa", "padding": "8px", "borderRadius": "4px"}))
        if isinstance(last_val, pd.DataFrame):
            children.append(
                dash_table.DataTable(
                    data=last_val.head(100).round(6).to_dict("records"),
                    columns=[{"name": c, "id": c} for c in last_val.columns],
                    page_size=15,
                    style_cell={"fontSize": "12px", "fontFamily": "monospace"},
                )
            )
        return children or html.Div("✓ Exécuté (pas de sortie)", className="text-muted")

    except Exception:
        return html.Pre(
            traceback.format_exc(),
            style={"color": "#dc3545", "fontSize": "12px", "backgroundColor": "#fff5f5", "padding": "8px", "borderRadius": "4px"},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Inspector disponible sur http://localhost:8051")
    app.run(debug=True, port=8051)
