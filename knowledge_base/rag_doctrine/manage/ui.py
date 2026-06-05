"""
ui.py — Onglet Dash "Doctrine RAG" pour canvas_app.py.

Composants :
  - dcc.Upload (drag-drop PDF/DOCX, multi-fichiers)
  - panneau gauche : liste des documents (cliquable)
  - panneau droit  : chunks (texte complet) du document sélectionné
  - bouton Refresh + Delete (avec confirmation)

Le pipeline d'ingestion réutilise cli.ingest_files (mêmes fonctions que la CLI).
"""
from __future__ import annotations

import base64
import logging
import tempfile
import traceback
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from . import cli as _cli
from . import _indexer

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

def doctrine_tab() -> list:
    """Retourne les enfants du dbc.Tab "Doctrine RAG"."""
    return [
        dcc.Store(id="doctrine-selected-doc", data=None),
        dcc.Store(id="doctrine-refresh-trigger", data=0),

        dbc.Container([
            # Header : upload + status
            dbc.Row([
                dbc.Col([
                    dcc.Upload(
                        id="doctrine-upload",
                        children=html.Div([
                            html.I(className="fa fa-cloud-upload-alt me-2"),
                            "Glisser-déposer PDF/DOCX ou ",
                            html.A("sélectionner"),
                        ]),
                        style={
                            "width": "100%", "height": "70px", "lineHeight": "70px",
                            "borderWidth": "2px", "borderStyle": "dashed",
                            "borderRadius": "5px", "textAlign": "center",
                            "borderColor": "#aaa", "color": "#555",
                            "backgroundColor": "#f9f9f9",
                        },
                        multiple=True,
                    ),
                ], width=9),
                dbc.Col([
                    dbc.Button(
                        [html.I(className="fa fa-sync me-1"), "Refresh"],
                        id="doctrine-btn-refresh", color="secondary", outline=True,
                        size="sm", className="mt-3",
                    ),
                ], width=3, className="text-end"),
            ], className="mb-2"),

            html.Div(id="doctrine-upload-status", className="mb-3"),

            # Split view : docs (gauche) + chunks (droite)
            dbc.Row([
                dbc.Col([
                    html.H5([
                        html.I(className="fa fa-folder-open me-2"),
                        "Documents",
                        html.Span(id="doctrine-docs-count", className="text-muted ms-2 small"),
                    ]),
                    html.Div(id="doctrine-docs-list", style={
                        "maxHeight": "70vh", "overflowY": "auto",
                        "borderRight": "1px solid #e0e0e0", "paddingRight": "10px",
                    }),
                ], width=4),
                dbc.Col([
                    html.Div(id="doctrine-chunks-panel", style={
                        "maxHeight": "75vh", "overflowY": "auto",
                        "paddingLeft": "15px",
                    }),
                ], width=8),
            ]),
        ], fluid=True, className="p-3"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de rendu
# ─────────────────────────────────────────────────────────────────────────────

def _render_docs_list(docs: list[dict], selected_doc_id: str | None) -> tuple[list, str]:
    if not docs:
        return [
            html.Div(
                "Aucun document indexé. Uploadez un PDF ou DOCX pour démarrer.",
                className="text-muted text-center mt-4",
            )
        ], ""
    items = []
    for d in docs:
        is_active = d["doc_id"] == selected_doc_id
        items.append(
            dbc.ListGroupItem(
                [
                    html.Div([
                        html.Strong(d["doc_id"], className="me-2"),
                        html.Span(f"{d['n_chunks']} ch.", className="badge bg-secondary"),
                    ], className="d-flex justify-content-between align-items-center"),
                    html.Div(
                        d["doc_title"][:80] + ("…" if len(d["doc_title"]) > 80 else ""),
                        className="small text-muted mt-1",
                    ),
                ],
                id={"type": "doctrine-doc-item", "doc_id": d["doc_id"]},
                action=True,
                active=is_active,
                className="mb-1 cursor-pointer",
            )
        )
    total_chunks = sum(d["n_chunks"] for d in docs)
    summary = f"({len(docs)} docs, {total_chunks} chunks)"
    return [dbc.ListGroup(items)], summary


def _render_chunks_panel(doc_id: str | None) -> list:
    if not doc_id:
        return [
            html.Div([
                html.I(className="fa fa-arrow-left me-2"),
                "Sélectionnez un document à gauche pour voir ses chunks.",
            ], className="text-muted text-center mt-5"),
        ]
    chunks = _cli.get_chunks_for_doc(doc_id)
    if not chunks:
        return [html.Div(f"Aucun chunk pour {doc_id}.", className="text-warning")]

    doc_title = chunks[0].get("doc_title", doc_id)
    md0 = chunks[0].get("metadata") or {}
    source = md0.get("source_filename", "—")
    ingested = md0.get("ingested_at", "—")

    items: list = [
        html.Div([
            html.H5([
                html.I(className="fa fa-file-alt me-2"),
                doc_title,
            ], className="mb-1"),
            html.Div([
                html.Span(f"{len(chunks)} chunks", className="badge bg-info me-2"),
                html.Span(source, className="text-muted small me-2"),
                html.Span(ingested, className="text-muted small"),
            ], className="mb-2"),
            dbc.Button(
                [html.I(className="fa fa-trash me-1"), f"Supprimer {doc_id}"],
                id="doctrine-btn-delete", color="danger", outline=True, size="sm",
                className="mb-3",
            ),
            html.Hr(),
        ]),
    ]

    for c in chunks:
        md = c.get("metadata") or {}
        tags = md.get("tags", [])
        regulatory = md.get("regulatory", False)
        has_formula = md.get("has_formula", False)
        word_count = c.get("word_count", 0)

        badges = []
        for t in tags:
            badges.append(html.Span(t, className="badge bg-primary me-1"))
        if regulatory:
            badges.append(html.Span("regulatory", className="badge bg-warning text-dark me-1"))
        if has_formula:
            badges.append(html.Span(f"{md.get('formula_count', 0)} formules", className="badge bg-success me-1"))
        badges.append(html.Span(f"{word_count} mots", className="badge bg-light text-dark me-1"))

        items.append(
            dbc.Card([
                dbc.CardHeader([
                    html.Strong(c.get("section_title", c.get("section_id", "?"))),
                ]),
                dbc.CardBody([
                    html.Div(badges, className="mb-2"),
                    html.Pre(
                        c.get("text", ""),
                        style={
                            "whiteSpace": "pre-wrap",
                            "fontFamily": "inherit",
                            "fontSize": "0.9rem",
                            "marginBottom": "0",
                        },
                    ),
                ]),
            ], className="mb-2")
        )
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Upload helper
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORTED_EXT = (".pdf", ".docx")


def _ingest_uploaded(contents_list: list[str], filenames: list[str]) -> tuple[str, str]:
    """Décode les fichiers uploadés en tempfiles puis appelle ingest_files.

    Retourne (color, message_html) pour l'alerte.
    """
    if not contents_list:
        return "warning", "Aucun fichier."

    accepted: list[Path] = []
    rejected: list[str] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="doctrine_upload_"))

    try:
        for content, name in zip(contents_list, filenames):
            path = tmp_dir / name
            ext = path.suffix.lower()
            if ext not in _SUPPORTED_EXT:
                rejected.append(f"{name} (extension non supportée)")
                continue
            try:
                _, b64 = content.split(",", 1)
                path.write_bytes(base64.b64decode(b64))
                accepted.append(path)
            except Exception as exc:
                rejected.append(f"{name} (décodage : {exc})")

        if not accepted:
            return "danger", "Aucun fichier valide. " + "; ".join(rejected)

        try:
            report = _cli.ingest_files(accepted, force=False, dry_run=False)
        except Exception as exc:
            log.error("ingest_files failed: %s\n%s", exc, traceback.format_exc())
            return "danger", f"Échec ingestion : {exc}"

        msgs: list[str] = []
        for a in report["added"]:
            msgs.append(f"✓ {a['file']} → {a['doc_id']} ({a['n_chunks']} chunks)")
        for s in report["skipped"]:
            msgs.append(f"⊘ {s['file']} : {s['reason']}")
        for r in rejected:
            msgs.append(f"✗ {r}")
        color = "success" if report["added"] else "warning"
        return color, " | ".join(msgs) or "Rien à faire"
    finally:
        # Cleanup tempdir
        try:
            for p in tmp_dir.iterdir():
                p.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Register callbacks
# ─────────────────────────────────────────────────────────────────────────────

def register_callbacks(app) -> None:
    """Enregistre les callbacks de l'onglet sur l'app Dash."""

    @app.callback(
        Output("doctrine-docs-list", "children"),
        Output("doctrine-docs-count", "children"),
        Input("doctrine-refresh-trigger", "data"),
        Input("doctrine-selected-doc", "data"),
    )
    def _cb_render_docs_list(_trigger, selected_doc_id):
        docs = _cli.list_docs()
        return _render_docs_list(docs, selected_doc_id)

    @app.callback(
        Output("doctrine-chunks-panel", "children"),
        Input("doctrine-selected-doc", "data"),
        Input("doctrine-refresh-trigger", "data"),
    )
    def _cb_render_chunks(selected_doc_id, _trigger):
        return _render_chunks_panel(selected_doc_id)

    @app.callback(
        Output("doctrine-selected-doc", "data"),
        Input({"type": "doctrine-doc-item", "doc_id": ALL}, "n_clicks"),
        State("doctrine-selected-doc", "data"),
        prevent_initial_call=True,
    )
    def _cb_select_doc(n_clicks_list, current):
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks_list):
            raise PreventUpdate
        trig = ctx.triggered[0]["prop_id"].split(".")[0]
        import json as _json
        try:
            parsed = _json.loads(trig)
            new_doc_id = parsed["doc_id"]
        except Exception:
            raise PreventUpdate
        # Toggle : re-click sur le doc actif → désélection
        if new_doc_id == current:
            return None
        return new_doc_id

    @app.callback(
        Output("doctrine-upload-status", "children"),
        Output("doctrine-refresh-trigger", "data", allow_duplicate=True),
        Input("doctrine-upload", "contents"),
        State("doctrine-upload", "filename"),
        State("doctrine-refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def _cb_upload(contents_list, filenames, trigger):
        if not contents_list:
            raise PreventUpdate
        # Dash peut passer une string si un seul fichier → on normalise en liste
        if isinstance(contents_list, str):
            contents_list = [contents_list]
            filenames = [filenames]
        color, msg = _ingest_uploaded(contents_list, filenames)
        alert = dbc.Alert(msg, color=color, dismissable=True, className="mb-0 py-2")
        return alert, (trigger or 0) + 1

    @app.callback(
        Output("doctrine-refresh-trigger", "data", allow_duplicate=True),
        Input("doctrine-btn-refresh", "n_clicks"),
        State("doctrine-refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def _cb_refresh(n, trigger):
        if not n:
            raise PreventUpdate
        return (trigger or 0) + 1

    @app.callback(
        Output("doctrine-upload-status", "children", allow_duplicate=True),
        Output("doctrine-selected-doc", "data", allow_duplicate=True),
        Output("doctrine-refresh-trigger", "data", allow_duplicate=True),
        Input("doctrine-btn-delete", "n_clicks"),
        State("doctrine-selected-doc", "data"),
        State("doctrine-refresh-trigger", "data"),
        prevent_initial_call=True,
    )
    def _cb_delete(n_clicks, selected_doc_id, trigger):
        if not n_clicks or not selected_doc_id:
            raise PreventUpdate
        try:
            rep = _cli.delete_doc(selected_doc_id)
        except Exception as exc:
            return dbc.Alert(f"Échec suppression : {exc}", color="danger", dismissable=True, className="mb-0 py-2"), selected_doc_id, trigger
        msg = f"✓ {rep['doc_id']} supprimé ({rep['deleted']} chunks). Restant : {rep['remaining']}."
        return dbc.Alert(msg, color="success", dismissable=True, className="mb-0 py-2"), None, (trigger or 0) + 1
