"""
encoder_app.py
Application Dash indépendante — Constructeur de prompt actuariel (Encodeur).

Lancer :  python encoder_app.py
URL :     http://localhost:8051

Workflow :
  1. L'utilisateur charge un rapport PDF de référence + souhaits complémentaires.
  2. GPT-4o analyse le PDF et génère la section MISSION (objectif, livrables, méthode).
  3. Le prompt final = MISSION + section technique de base (bibliothèque de fonctions).
  4. (Optionnel) Boucle d'optimisation : l'agent actuariel tourne sur données synthétiques,
     le LLM-as-judge évalue la structure du rapport produit, le prompt est affiné
     jusqu'à atteindre le seuil de score fixé par l'utilisateur.
  5. L'utilisateur télécharge le template.json et le charge dans canvas_app.py (décodeur).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
from pathlib import Path

# Add project root (for agent.py) and agent_builder/ (for judge_agent.py) to path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback_context, dcc, html
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
_UPLOADS_DIR = _ROOT / "uploads"
_UPLOADS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# État global de l'encodeur (thread-safe)
# ─────────────────────────────────────────────────────────────────────────────
_enc_results: dict = {
    "status": "idle",          # idle | analyzing | running | done | error
    "progress": "",            # message de progression
    "template": None,          # dict template structuré
    "mission": "",             # prompt rédacteur (agent_system_prompt) — writer prompt UNIQUEMENT
    "history": [],             # [{iter, score, scores, ecarts}]
    "error": "",               # message d'erreur
    # Simulation
    "sim_status": "idle",      # idle | running | done | error
    "sim_progress": "",        # message de progression simulation
    "sim_pdf_bytes": None,     # bytes du PDF simulé
}
_enc_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Analyse (thread)
# ─────────────────────────────────────────────────────────────────────────────

def _run_analysis_thread(pdf_bytes: bytes, filename: str, wishes: str) -> None:
    """Analyse le PDF en background."""
    from analyze_report_template import analyze_report_pdf

    def progress(msg: str):
        with _enc_lock:
            _enc_results["progress"] = msg

    with _enc_lock:
        _enc_results["status"] = "analyzing"
        _enc_results["progress"] = "Extraction du texte PDF…"
        _enc_results["template"] = None
        _enc_results["mission"] = ""
        _enc_results["history"] = []
        _enc_results["error"] = ""
        _enc_results["sim_status"] = "idle"
        _enc_results["sim_progress"] = ""
        _enc_results["sim_pdf_bytes"] = None

    try:
        template = analyze_report_pdf(
            pdf_bytes=pdf_bytes,
            filename=filename,
            additional_wishes=wishes,
            progress_fn=progress,
        )
        mission = template.get("agent_system_prompt", "")

        with _enc_lock:
            _enc_results["template"] = template
            _enc_results["mission"] = mission
            _enc_results["status"] = "done"
            _enc_results["progress"] = "Analyse terminée."

    except Exception as exc:
        with _enc_lock:
            _enc_results["status"] = "error"
            _enc_results["error"] = str(exc)
            _enc_results["progress"] = f"Erreur : {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Boucle d'optimisation (thread)
# ─────────────────────────────────────────────────────────────────────────────

def _run_loop_thread(threshold: float, max_iter: int = 3) -> None:
    """Boucle d'optimisation : agent → judge → affiner la section MISSION."""
    from agent import make_isolated_kernel, run_agent_on_synthetic
    from judge_agent import evaluate_report_structure

    with _enc_lock:
        template = dict(_enc_results.get("template") or {})
        mission = _enc_results.get("mission", "")

    if not template or not mission:
        with _enc_lock:
            _enc_results["status"] = "error"
            _enc_results["error"] = "Aucun template — lancez d'abord l'analyse."
        return

    with _enc_lock:
        _enc_results["status"] = "running"
        _enc_results["history"] = []
        _enc_results["progress"] = "Démarrage de la boucle d'optimisation…"

    best_score = 0.0
    best_mission = mission
    current_mission = mission  # version en cours de raffinement
    regressions = 0

    for i in range(max_iter):
        with _enc_lock:
            _enc_results["progress"] = f"Itération {i + 1}/{max_iter} — exécution de l'agent…"

        try:
            from agent import SYSTEM_PROMPT_TEMPLATE
            kernel = make_isolated_kernel()
            steps, summary = run_agent_on_synthetic(SYSTEM_PROMPT_TEMPLATE, kernel=kernel)
        except Exception as exc:
            with _enc_lock:
                _enc_results["progress"] = f"Erreur agent (iter {i + 1}) : {exc}"
            regressions += 1
            if regressions >= 2:
                break
            continue

        with _enc_lock:
            _enc_results["progress"] = f"Itération {i + 1}/{max_iter} — évaluation LLM-as-judge…"

        try:
            result = evaluate_report_structure(template, steps, summary)
        except Exception as exc:
            with _enc_lock:
                _enc_results["progress"] = f"Erreur judge (iter {i + 1}) : {exc}"
            regressions += 1
            if regressions >= 2:
                break
            continue

        score = result["score_global"]
        iter_record = {
            "iter": i + 1,
            "score": score,
            "scores": result["scores"],
            "ecarts": result["ecarts"][:5],
            "verdict": result["verdict"],
        }

        with _enc_lock:
            _enc_results["history"].append(iter_record)

        if score > best_score:
            best_score = score
            best_mission = current_mission  # sauvegarder la version courante raffinée
            regressions = 0
        else:
            regressions += 1
            if regressions >= 2:
                with _enc_lock:
                    _enc_results["progress"] = (
                        f"Arrêt : 2 itérations sans amélioration (score = {best_score:.2f})."
                    )
                break

        if score >= threshold:
            with _enc_lock:
                _enc_results["progress"] = (
                    f"Seuil atteint : score = {score:.2f} ≥ {threshold:.2f}."
                )
            break

        # Affiner current_mission (pas best_mission) pour l'itération suivante
        current_mission = _refine_mission(current_mission, result["ecarts"], result["suggestions"])

    with _enc_lock:
        _enc_results["mission"] = best_mission
        if _enc_results.get("template"):
            _enc_results["template"]["agent_system_prompt"] = best_mission
        _enc_results["status"] = "done"
        if not _enc_results["progress"].startswith(("Seuil", "Arrêt")):
            _enc_results["progress"] = f"Boucle terminée — meilleur score : {best_score:.2f}"


def _refine_mission(mission: str, ecarts: list[str], suggestions: list[str]) -> str:
    """Affine la section MISSION via GPT-4o en se basant sur les écarts et suggestions."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return mission

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        ecarts_str = "\n".join(f"- {e}" for e in ecarts[:8])
        sug_str = "\n".join(f"- {s}" for s in suggestions[:4])

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un expert en ingénierie de prompts actuariels. "
                        "On te fournit la section MISSION actuelle d'un prompt et les écarts "
                        "détectés par un évaluateur automatique. "
                        "Retourne UNIQUEMENT la section MISSION améliorée, sans texte autour. "
                        "Conserve exactement la structure (OBJECTIF, LIVRABLES OBLIGATOIRES, "
                        "MÉTHODE IMPOSÉE, EXIGENCES COMPLÉMENTAIRES, FORMAT DU RÉSUMÉ FINAL). "
                        "Corrige les manques pointés par les écarts. N'allonge pas inutilement."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"SECTION MISSION ACTUELLE :\n{mission}\n\n"
                        f"ÉCARTS DÉTECTÉS :\n{ecarts_str}\n\n"
                        f"SUGGESTIONS D'AMÉLIORATION :\n{sug_str}\n\n"
                        "Retourne la section MISSION améliorée."
                    ),
                },
            ],
            max_tokens=2000,
            temperature=0.3,
        )
        return (resp.choices[0].message.content or mission).strip()
    except Exception:
        return mission


# ─────────────────────────────────────────────────────────────────────────────
# Simulation — génère un rapport de démo avec données synthétiques
# ─────────────────────────────────────────────────────────────────────────────

def _infer_domain_label(template: dict) -> str:
    """Déduit le domain_label à partir du titre et de la méthodologie du template."""
    title = template.get("report_title", "").lower()
    if any(k in title for k in ["mortalité", "mortalite", "décès", "deces", "table"]):
        return "mortality"
    if any(k in title for k in ["ibnr", "provision", "non-vie", "sinistre"]):
        return "nonlife_reserving"
    if any(k in title for k in ["vif", "vie", "contrat", "valuation"]):
        return "life_valuation"
    return "mortality"


def _generate_synthetic_context(mission: str, template: dict) -> tuple[list[dict], str]:
    """Génère des steps et un summary synthétiques réalistes via LLM."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return [], "Aucun résultat synthétique disponible (clé API manquante)."

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    methodology = template.get("methodology", {})
    meth_desc = (
        f"Méthode : {methodology.get('smoother', 'Whittaker-Henderson')}, "
        f"table de référence : {methodology.get('reference_table', 'TH0002')}, "
        f"âges {methodology.get('age_min', 20)}-{methodology.get('age_max', 90)}, "
        f"segmentation : {methodology.get('segmentation', 'par sexe')}"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un actuaire. Génère des résultats de calcul synthétiques RÉALISTES "
                    "qui correspondraient aux variables déclarées dans le prompt rédacteur ci-dessous.\n"
                    "Retourne un JSON STRICT avec :\n"
                    "- \"summary\" (str) : synthèse narrative de l'analyse (300-500 mots) avec "
                    "  valeurs numériques typiques (SMR ~0.90-1.10, chi2, p-value, lambda, etc.)\n"
                    "- \"display_outputs\" (list[str]) : 4-6 tableaux ASCII réalistes correspondant "
                    "  aux DataFrames déclarés dans le prompt (en-tête + 5-8 lignes de données).\n"
                    "Utilise des noms de colonnes cohérents avec le prompt rédacteur."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Méthodologie : {meth_desc}\n\n"
                    f"Prompt rédacteur (variables attendues) :\n{mission[:3500]}"
                ),
            },
        ],
        max_tokens=2500,
        temperature=0.4,
        response_format={"type": "json_object"},
    )

    raw = (resp.choices[0].message.content or "{}").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], "Données synthétiques non disponibles (erreur JSON)."

    summary = data.get("summary", "")
    display_outputs = data.get("display_outputs", [])

    steps = []
    for i, table_text in enumerate(display_outputs):
        label = f"Tableau synthétique {i + 1}"
        steps.append({
            "description": f"Résultats synthétiques ({i + 1}/{len(display_outputs)})",
            "output": str(table_text),
            "figures": [],
            "display_outputs": [{"text": str(table_text), "label": label}],
        })

    return steps, summary


def _run_simulation_thread() -> None:
    """Lance une simulation complète : données synthétiques → report_agent → PDF."""
    import tempfile
    from pathlib import Path as _Path
    from report_agent.generate_report import generate_mortality_report as _gen_mortality_report
    from report_payload_builder import build_report_payload as _build_report_payload, build_exposure_deciles as _bde

    with _enc_lock:
        template = dict(_enc_results.get("template") or {})
        mission = _enc_results.get("mission", "")
        _enc_results["sim_status"] = "running"
        _enc_results["sim_progress"] = "Génération des données synthétiques…"
        _enc_results["sim_pdf_bytes"] = None

    if not mission:
        with _enc_lock:
            _enc_results["sim_status"] = "error"
            _enc_results["sim_progress"] = "Aucun prompt — lancez d'abord l'analyse."
        return

    try:
        steps, summary = _generate_synthetic_context(mission, template)

        with _enc_lock:
            _enc_results["sim_progress"] = (
                f"Données synthétiques générées ({len(steps)} tables) — génération des graphiques…"
            )

        methodology = template.get("methodology", {})
        domain_label = _infer_domain_label(template)
        report_title = template.get("report_title", "Rapport actuariel")

        with _enc_lock:
            _enc_results["sim_progress"] = "Génération des données synthétiques et du rapport…"

        # Données synthétiques Gompertz-Makeham
        import numpy as _np
        _rng = _np.random.default_rng(42)
        _age_min = int((methodology or {}).get("age_min", 25))
        _age_max = int((methodology or {}).get("age_max", 85))
        _ages  = _np.arange(_age_min, _age_max + 1, dtype=float)
        _n     = len(_ages)
        _A, _B, _c = 0.0003, 0.000025, 0.10
        _q_ref = _np.clip(_A + _B * _np.exp(_c * (_ages - 50)), 5e-5, 0.9)
        _exp   = _np.maximum(
            _rng.integers(300, 3500, _n).astype(float) * _np.linspace(1.0, 0.25, _n),
            20.0,
        )
        _deaths = _rng.poisson(_q_ref * _exp).astype(float)
        _q_brut = _np.where(_exp > 0, _deaths / _exp, _q_ref)
        _kernel  = _np.exp(-0.5 * (_np.arange(-4, 5, dtype=float) / 2.0) ** 2)
        _kernel /= _kernel.sum()
        _q_lisse = _np.maximum(_np.convolve(_q_brut, _kernel, mode="same"), 1e-5)
        _ic_inf  = _np.maximum(0.0, _q_lisse - 1.96 * _np.sqrt(_q_lisse / _np.maximum(_exp, 1.0)))
        _ic_sup  = _q_lisse + 1.96 * _np.sqrt(_q_lisse / _np.maximum(_exp, 1.0))
        _D_obs   = float(_deaths.sum())
        _D_exp   = float((_q_ref * _exp).sum())
        _smr     = _D_obs / _D_exp
        _D_i     = max(_D_obs, 1.0)
        _smr_lo  = (_D_i/_D_exp)*(1-1/(9*_D_i)-1.96/(3*_np.sqrt(_D_i)))**3
        _smr_hi  = ((_D_i+1)/_D_exp)*(1-1/(9*(_D_i+1))+1.96/(3*_np.sqrt(_D_i+1)))**3
        _chi2    = float(_np.sum((_deaths - _q_ref*_exp)**2 / _np.maximum(_q_ref*_exp, 1.0)))
        _ddl     = max(_n - 2, 1)
        try:
            from scipy.stats import chi2 as _chi2d
            _pval = float(1 - _chi2d.cdf(_chi2, _ddl))
        except Exception:
            _pval = 0.05
        _abat = float(_np.sum(_q_lisse * _exp) / max(float(_np.sum(_q_ref * _exp)), 1e-10))

        _portfolio = {
            "n_assures": 10000, "n_contrats_actifs": 10000,
            "type_contrat": domain_label or "vie_entiere",
            "periode_debut": "2010-01-01", "periode_fin": "2023-12-31",
            "age_min": _age_min, "age_max": _age_max,
            "segmentation": "global", "table_reference": "TH00-02",
        }
        _qualite = {
            "traitements_appliques": [
                {"nom": "Simulation", "description": f"Données synthétiques — {report_title}"},
            ],
            "stats_annuelles": [
                {"annee": yr, "exposition": int(_exp.sum() / 14),
                 "age_moyen": round(float(_np.average(_ages, weights=_exp)), 1),
                 "deces": int(_deaths.sum() / 14)}
                for yr in range(2010, 2024)
            ],
        }

        _mortality_payload = _build_report_payload(
            ages=_ages, exposure=_exp, deaths_observed=_deaths,
            q_brut=_q_brut, q_lisse=_q_lisse, ic_inf=_ic_inf, ic_sup=_ic_sup,
            q_ref=_q_ref, methode="whittaker_henderson",
            parametres={"lambda": 1000, "ordre": 2},
            smr_global=_smr, smr_ic_inf=_smr_lo, smr_ic_sup=_smr_hi,
            chi2_stat=_chi2, chi2_ddl=_ddl, chi2_pvalue=_pval,
            abattement_global=_abat,
            portfolio_info=_portfolio, qualite_info=_qualite,
            trace_info={"study_ref": "SIMULATION", "writer_prompt_len": len(mission)},
        )

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp_path = f.name
        _gen_mortality_report(_mortality_payload, output_path=tmp_path)

        pdf_bytes = _Path(tmp_path).read_bytes()
        try:
            _Path(tmp_path).unlink()
        except OSError:
            pass

        with _enc_lock:
            _enc_results["sim_status"] = "done"
            _enc_results["sim_progress"] = f"Rapport simulé prêt ({len(pdf_bytes) // 1024} Ko)."
            _enc_results["sim_pdf_bytes"] = pdf_bytes

    except Exception as exc:
        with _enc_lock:
            _enc_results["sim_status"] = "error"
            _enc_results["sim_progress"] = f"Erreur simulation : {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — rendu de la structure extraite
# ─────────────────────────────────────────────────────────────────────────────

def _render_structure(template: dict) -> list:
    """Construit les composants Dash pour afficher la structure extraite."""
    if not template:
        return [html.P("Aucun template chargé.", style={"color": "#999"})]

    items = []

    title = template.get("report_title", "?")
    items.append(html.H5(title, style={"color": "#2D2D2D", "fontWeight": "bold"}))

    source = template.get("source_pdf", "")
    if source:
        items.append(html.P(f"Source : {source}", style={"color": "#777", "fontSize": "12px"}))

    # Sections
    sections = template.get("sections", [])
    if sections:
        items.append(html.H6("Sections", style={"marginTop": "10px", "color": "#555"}))
        for s in sections:
            items.append(html.Li(
                f"{s.get('id', '?')} — {s.get('title', '?')}",
                style={"fontSize": "12px"},
            ))

    # Tableaux
    tables = template.get("tables", [])
    if tables:
        items.append(html.H6("Tableaux", style={"marginTop": "10px", "color": "#555"}))
        for t in tables:
            cols = ", ".join(t.get("columns", [])[:5])
            items.append(html.Li(
                [
                    html.Strong(f"{t.get('id', '?')} — {t.get('name', '?')}",
                                style={"fontSize": "12px"}),
                    html.Br(),
                    html.Span(f"Colonnes : {cols}", style={"fontSize": "11px", "color": "#777"}),
                ],
                style={"marginBottom": "4px"},
            ))

    # Figures
    figures = template.get("figures", [])
    if figures:
        items.append(html.H6("Graphiques", style={"marginTop": "10px", "color": "#555"}))
        for f in figures:
            items.append(html.Li(
                [
                    html.Strong(f"{f.get('id', '?')} — {f.get('title', '?')}",
                                style={"fontSize": "12px"}),
                    html.Br(),
                    html.Span(
                        f"x={f.get('x_axis', '?')}, y={f.get('y_axis', '?')} "
                        f"→ {f.get('python_function', '?')}",
                        style={"fontSize": "11px", "color": "#777"},
                    ),
                ],
                style={"marginBottom": "4px"},
            ))

    # Méthode
    meth = template.get("methodology", {})
    if meth:
        items.append(html.H6("Méthode", style={"marginTop": "10px", "color": "#555"}))
        items.append(html.Ul([
            html.Li(f"Lissage : {meth.get('smoother', '?')}"
                    + (f" λ={meth['lambda']}" if meth.get("lambda") else ""),
                    style={"fontSize": "12px"}),
            html.Li(f"Table de référence : {meth.get('reference_table', '?')}",
                    style={"fontSize": "12px"}),
            html.Li(f"Plage d'âges : {meth.get('age_min', '?')}–{meth.get('age_max', '?')} ans",
                    style={"fontSize": "12px"}),
            html.Li(f"Découpage : {meth.get('segmentation', '?')}",
                    style={"fontSize": "12px"}),
        ]))

    return items


def _render_sim_panel(sim_status: str, sim_progress: str) -> list:
    """Construit le panneau d'état de la simulation."""
    if sim_status == "idle":
        return [html.P(
            "Cliquez sur '▶ Simuler la rédaction' pour générer un rapport de démonstration "
            "avec des données synthétiques réalistes.",
            style={"color": "#999", "textAlign": "center", "marginTop": "40px"},
        )]
    if sim_status == "running":
        return [
            html.Div([
                dbc.Spinner(size="sm", color="info"),
                html.Span(f" {sim_progress}", style={"fontSize": "12px", "color": "#555",
                                                      "marginLeft": "8px"}),
            ], style={"marginTop": "40px", "textAlign": "center"}),
        ]
    if sim_status == "error":
        return [html.P(f"❌ {sim_progress}",
                       style={"color": "#C0392B", "fontSize": "12px", "padding": "8px"})]
    # done
    return [
        dbc.Alert([
            html.Strong("Rapport simulé prêt. "),
            sim_progress,
            html.Br(),
            html.Small(
                "Le rapport a été généré avec des données synthétiques correspondant "
                "aux variables déclarées dans le prompt rédacteur.",
                style={"color": "#555"},
            ),
        ], color="success", style={"fontSize": "12px", "marginTop": "20px"}),
    ]


def _render_history(history: list) -> list:
    """Construit les composants Dash pour afficher l'historique des itérations."""
    if not history:
        return [html.P("Aucune itération encore.", style={"color": "#999", "fontSize": "12px"})]

    rows = []
    for h in history:
        score = h.get("score", 0)
        color = "#4CAF50" if score >= 0.80 else "#FF9800" if score >= 0.60 else "#F44336"
        rows.append(
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            html.Strong(f"Iter {h['iter']} — Score : {score:.2f}",
                                        style={"color": color, "fontSize": "13px"}),
                            width=8,
                        ),
                        dbc.Col(
                            dbc.Badge(h.get("verdict", "?")[:40],
                                      color="success" if score >= 0.80 else "warning",
                                      style={"fontSize": "9px"}),
                            width=4,
                        ),
                    ]),
                    html.Hr(style={"margin": "6px 0"}),
                    *[
                        html.P(f"• {e}", style={"fontSize": "11px", "color": "#555", "margin": "2px 0"})
                        for e in h.get("ecarts", [])[:3]
                    ],
                ], style={"padding": "8px"}),
                style={"marginBottom": "8px", "background": "#F5F2E7",
                       "border": "1px solid #D8D0C4"},
            )
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

_LABEL_STYLE = {"fontSize": "11px", "color": "#777", "marginBottom": "3px"}
_CARD_STYLE = {
    "background": "#FBF8F1", "border": "1px solid #D8D0C4",
    "borderRadius": "8px", "padding": "12px", "marginBottom": "10px",
}

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Encodeur Actuariel",
)

app.layout = dbc.Container(
    [
        # ── Header ──────────────────────────────────────────────────────────
        dbc.Navbar(
            dbc.Container(
                [
                    html.Span("🎯 Constructeur de prompt actuariel",
                              className="navbar-brand",
                              style={"fontWeight": "bold", "fontSize": "16px"}),
                    html.Span("Encodeur — génère le prompt MISSION pour l'agent actuariel",
                              style={"color": "#aaa", "fontSize": "12px"}),
                ],
                fluid=True,
            ),
            color="light",
            dark=False,
            style={"borderBottom": "1px solid #C5BDB0", "background": "#F0EDE3",
                   "padding": "6px 16px", "marginBottom": "16px"},
        ),

        dbc.Row([
            # ── Colonne gauche — inputs ──────────────────────────────────────
            dbc.Col([
                # Upload PDF
                html.Div([
                    html.P("Rapport PDF de référence", style=_LABEL_STYLE),
                    dcc.Upload(
                        id="enc-upload-pdf",
                        children=html.Div([
                            "📄 Charger un PDF",
                            html.Br(),
                            html.Small("(glisser-déposer)", style={"color": "#888"}),
                        ]),
                        multiple=False, accept=".pdf",
                        style={
                            "border": "2px dashed #A09890", "borderRadius": "8px",
                            "padding": "12px", "textAlign": "center",
                            "color": "#555", "fontSize": "12px", "cursor": "pointer",
                        },
                    ),
                    html.Div(id="enc-pdf-filename",
                             style={"color": "#4CAF50", "fontSize": "11px",
                                    "minHeight": "16px", "marginTop": "4px"}),
                ], style=_CARD_STYLE),

                # Souhaits complémentaires
                html.Div([
                    html.P("Souhaits complémentaires (optionnel)", style=_LABEL_STYLE),
                    dcc.Textarea(
                        id="enc-wishes",
                        placeholder=(
                            "Ex :\n"
                            "- Ajouter une analyse par tranche d'âge quinquennale\n"
                            "- Comparer avec la table TF0002\n"
                            "- Inclure un graphique de survie"
                        ),
                        rows=5,
                        style={"width": "100%", "fontSize": "12px", "resize": "vertical",
                               "borderRadius": "4px", "border": "1px solid #C5BDB0"},
                    ),
                ], style=_CARD_STYLE),

                # Bouton analyser
                dbc.Button(
                    "🔍 Générer le prompt",
                    id="enc-btn-generate",
                    color="primary", size="sm",
                    className="w-100 mb-2",
                    disabled=True,
                ),

                html.Hr(style={"borderColor": "#C5BDB0"}),

                # Boucle optionnelle
                html.P("Boucle d'optimisation (optionnel)", style=_LABEL_STYLE),
                dbc.Checklist(
                    id="enc-loop-toggle",
                    options=[{"label": " Activer la boucle", "value": "on"}],
                    value=[],
                    style={"fontSize": "12px"},
                ),
                html.Div([
                    html.P("Seuil de score cible (0.5 → 1.0)",
                           style={**_LABEL_STYLE, "marginTop": "8px"}),
                    dcc.Slider(
                        id="enc-threshold",
                        min=0.5, max=1.0, step=0.05, value=0.80,
                        marks={0.5: "0.5", 0.7: "0.7", 0.85: "0.85", 1.0: "1.0"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    html.P("Itérations max",
                           style={**_LABEL_STYLE, "marginTop": "8px"}),
                    dcc.Slider(
                        id="enc-max-iter",
                        min=1, max=5, step=1, value=3,
                        marks={1: "1", 2: "2", 3: "3", 4: "4", 5: "5"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], id="enc-loop-config", style={"display": "none"}),

                dbc.Button(
                    "▶ Lancer la boucle",
                    id="enc-btn-loop",
                    color="warning", size="sm",
                    className="w-100 mb-2 mt-2",
                    disabled=True,
                ),

                html.Hr(style={"borderColor": "#C5BDB0"}),

                # Simulation
                html.P("Simulation rédacteur", style=_LABEL_STYLE),
                html.P(
                    "Génère un rapport complet avec des données synthétiques "
                    "pour vérifier que le prompt atteint la cible.",
                    style={"fontSize": "11px", "color": "#888", "marginBottom": "6px"},
                ),
                dbc.Button(
                    "▶ Simuler la rédaction",
                    id="enc-btn-simulate",
                    color="info", size="sm",
                    className="w-100 mb-2",
                    disabled=True,
                ),

                # Statut
                html.Div(
                    id="enc-status",
                    style={"fontSize": "11px", "color": "#777",
                           "minHeight": "30px", "marginTop": "4px"},
                ),

            ], width=4, style={"padding": "0 12px"}),

            # ── Colonne droite — résultats ────────────────────────────────────
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(
                        html.Div(
                            id="enc-structure",
                            style={"overflowY": "auto", "height": "calc(100vh - 180px)",
                                   "padding": "8px"},
                            children=[html.P("Chargez un PDF et cliquez sur 'Générer le prompt'.",
                                             style={"color": "#999", "textAlign": "center",
                                                    "marginTop": "40px"})],
                        ),
                        label="Structure extraite",
                        tab_id="t-structure",
                    ),
                    dbc.Tab(
                        dcc.Textarea(
                            id="enc-prompt-display",
                            rows=30,
                            readOnly=False,
                            style={"width": "100%", "fontFamily": "monospace",
                                   "fontSize": "11px", "resize": "vertical",
                                   "height": "calc(100vh - 180px)"},
                        ),
                        label="Prompt généré",
                        tab_id="t-prompt",
                    ),
                    dbc.Tab(
                        html.Div(
                            id="enc-score-panel",
                            style={"overflowY": "auto", "height": "calc(100vh - 230px)",
                                   "padding": "8px"},
                            children=[html.P("Activez la boucle d'optimisation pour voir les scores.",
                                             style={"color": "#999", "textAlign": "center",
                                                    "marginTop": "40px"})],
                        ),
                        label="Score & itérations",
                        tab_id="t-score",
                    ),
                    dbc.Tab(
                        html.Div([
                            html.Div(
                                id="enc-sim-panel",
                                style={"overflowY": "auto", "padding": "8px"},
                                children=[
                                    html.P(
                                        "Cliquez sur '▶ Simuler la rédaction' pour générer "
                                        "un rapport de démonstration avec des données synthétiques.",
                                        style={"color": "#999", "textAlign": "center",
                                               "marginTop": "40px"},
                                    )
                                ],
                            ),
                            dbc.Button(
                                "⬇ Télécharger le rapport simulé (PDF)",
                                id="enc-btn-download-sim",
                                color="success", size="sm",
                                className="mt-2",
                                disabled=True,
                            ),
                            dcc.Download(id="enc-download-sim"),
                        ], style={"height": "calc(100vh - 230px)", "display": "flex",
                                  "flexDirection": "column", "padding": "8px"}),
                        label="Rapport simulé",
                        tab_id="t-simulate",
                    ),
                ], id="enc-tabs", active_tab="t-structure"),

                # Bouton télécharger
                dbc.Button(
                    "💾 Télécharger template.json",
                    id="enc-btn-download",
                    color="success", size="sm",
                    className="mt-2",
                    disabled=True,
                ),
                dcc.Download(id="enc-download"),
            ], width=8, style={"padding": "0 12px"}),
        ]),

        # Stores et interval
        dcc.Store(id="enc-pdf-store", data=None),   # {bytes_b64, filename}
        dcc.Interval(id="enc-interval", interval=1200, disabled=True, n_intervals=0),
    ],
    fluid=True,
    style={"background": "#FBF8F1", "minHeight": "100vh", "padding": "0"},
)


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("enc-pdf-store", "data"),
    Output("enc-pdf-filename", "children"),
    Output("enc-btn-generate", "disabled"),
    Input("enc-upload-pdf", "contents"),
    State("enc-upload-pdf", "filename"),
    prevent_initial_call=True,
)
def handle_pdf_upload(contents, filename):
    if not contents:
        return None, "", True
    return {"b64": contents, "filename": filename or "rapport.pdf"}, f"✓ {filename}", False


@app.callback(
    Output("enc-loop-config", "style"),
    Input("enc-loop-toggle", "value"),
    prevent_initial_call=False,
)
def toggle_loop_config(values):
    if "on" in (values or []):
        return {"display": "block"}
    return {"display": "none"}


@app.callback(
    Output("enc-interval", "disabled"),
    Output("enc-status", "children"),
    Input("enc-btn-generate", "n_clicks"),
    State("enc-pdf-store", "data"),
    State("enc-wishes", "value"),
    prevent_initial_call=True,
)
def start_analysis(n_clicks, pdf_store, wishes):
    if not pdf_store:
        return True, "⚠ Chargez d'abord un PDF."
    try:
        _, b64 = pdf_store["b64"].split(",", 1)
        pdf_bytes = base64.b64decode(b64)
    except Exception as exc:
        return True, f"❌ Erreur lecture PDF : {exc}"

    t = threading.Thread(
        target=_run_analysis_thread,
        args=(pdf_bytes, pdf_store.get("filename", "rapport.pdf"), wishes or ""),
        daemon=True,
    )
    t.start()
    return False, "⏳ Analyse en cours…"


@app.callback(
    Output("enc-interval", "disabled", allow_duplicate=True),
    Output("enc-status", "children", allow_duplicate=True),
    Input("enc-btn-loop", "n_clicks"),
    State("enc-threshold", "value"),
    State("enc-max-iter", "value"),
    prevent_initial_call=True,
)
def start_loop(n_clicks, threshold, max_iter):
    with _enc_lock:
        status = _enc_results.get("status", "idle")
        template = _enc_results.get("template")

    if not template:
        return True, "⚠ Lancez d'abord l'analyse du PDF."
    if status == "running":
        return False, "⏳ Boucle déjà en cours…"

    t = threading.Thread(
        target=_run_loop_thread,
        args=(float(threshold or 0.80), int(max_iter or 3)),
        daemon=True,
    )
    t.start()
    return False, "⏳ Boucle d'optimisation démarrée…"


@app.callback(
    Output("enc-structure", "children"),
    Output("enc-prompt-display", "value"),
    Output("enc-score-panel", "children"),
    Output("enc-sim-panel", "children"),
    Output("enc-interval", "disabled", allow_duplicate=True),
    Output("enc-status", "children", allow_duplicate=True),
    Output("enc-btn-loop", "disabled"),
    Output("enc-btn-download", "disabled"),
    Output("enc-btn-simulate", "disabled"),
    Output("enc-btn-download-sim", "disabled"),
    Input("enc-interval", "n_intervals"),
    prevent_initial_call=True,
)
def refresh_encoder(n):
    with _enc_lock:
        results = dict(_enc_results)

    status = results.get("status", "idle")
    progress = results.get("progress", "")
    template = results.get("template")
    mission = results.get("mission", "")
    history = results.get("history", [])
    error = results.get("error", "")
    sim_status = results.get("sim_status", "idle")
    sim_progress = results.get("sim_progress", "")

    done = status in ("done", "error", "idle")
    sim_done = sim_status in ("done", "error", "idle")
    has_template = template is not None
    sim_pdf_ready = results.get("sim_pdf_bytes") is not None

    # Structure extraite
    structure_children = _render_structure(template) if template else [
        html.P(progress or "En attente…",
               style={"color": "#777", "fontSize": "12px", "textAlign": "center",
                      "marginTop": "40px"}),
    ]

    # Prompt rédacteur uniquement — jamais la BIBLIOTHÈQUE de l'agent de calcul
    prompt_display = mission

    # Score panel
    score_children = _render_history(history)

    # Simulation panel
    sim_children = _render_sim_panel(sim_status, sim_progress)

    # Status message
    if status == "error":
        status_msg = f"❌ {error}"
    elif sim_status == "running":
        status_msg = f"⏳ Simulation : {sim_progress}"
    elif sim_status == "done":
        status_msg = f"✓ {sim_progress}"
    elif sim_status == "error":
        status_msg = f"❌ Simulation — {sim_progress}"
    elif status == "done" and not history:
        status_msg = "✓ Analyse terminée. Téléchargez le template ou activez la boucle."
    elif status == "done":
        best = max((h["score"] for h in history), default=0)
        status_msg = f"✓ Boucle terminée — meilleur score : {best:.2f}"
    else:
        status_msg = progress or "⏳ En cours…"

    busy = status == "running" or sim_status == "running"

    return (
        structure_children,
        prompt_display,
        score_children,
        sim_children,
        done and sim_done,                          # interval disabled
        status_msg,
        not has_template or busy,                  # btn-loop disabled
        not has_template or busy,                  # btn-download disabled
        not has_template or busy,                  # btn-simulate disabled
        not sim_pdf_ready,                         # btn-download-sim disabled
    )


@app.callback(
    Output("enc-interval", "disabled", allow_duplicate=True),
    Output("enc-status", "children", allow_duplicate=True),
    Output("enc-tabs", "active_tab"),
    Input("enc-btn-simulate", "n_clicks"),
    prevent_initial_call=True,
)
def start_simulation(n_clicks):
    with _enc_lock:
        template = _enc_results.get("template")
        mission = _enc_results.get("mission", "")
        sim_status = _enc_results.get("sim_status", "idle")

    if not template or not mission:
        return True, "⚠ Lancez d'abord l'analyse du PDF.", "t-simulate"
    if sim_status == "running":
        return False, "⏳ Simulation déjà en cours…", "t-simulate"

    t = threading.Thread(target=_run_simulation_thread, daemon=True)
    t.start()
    return False, "⏳ Simulation démarrée…", "t-simulate"


@app.callback(
    Output("enc-download-sim", "data"),
    Input("enc-btn-download-sim", "n_clicks"),
    prevent_initial_call=True,
)
def download_simulation_pdf(n_clicks):
    with _enc_lock:
        pdf_bytes = _enc_results.get("sim_pdf_bytes")
        template = _enc_results.get("template") or {}
    if not pdf_bytes:
        return dash.no_update
    filename = (
        template.get("report_title", "rapport_simule")
        .replace(" ", "_")
        .replace("/", "-")[:40]
    ) + "_simulation.pdf"
    return dcc.send_bytes(pdf_bytes, filename, type="application/pdf")


@app.callback(
    Output("enc-download", "data"),
    Input("enc-btn-download", "n_clicks"),
    prevent_initial_call=True,
)
def download_template(n_clicks):
    with _enc_lock:
        template = _enc_results.get("template")
    if not template:
        return dash.no_update
    filename = (
        template.get("report_title", "template")
        .replace(" ", "_")
        .replace("/", "-")[:40]
    ) + "_template.json"
    # Exclure prompt_final s'il existe (résidu d'une ancienne version)
    template_clean = {k: v for k, v in template.items() if k != "prompt_final"}
    content = json.dumps(template_clean, ensure_ascii=False, indent=2)
    return dcc.send_string(content, filename, type="application/json")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Encodeur Actuariel — http://localhost:8051")
    print("Chargez un PDF de référence pour générer le prompt MISSION.")
    print("=" * 60)
    app.run(debug=False, port=8051, host="0.0.0.0")
