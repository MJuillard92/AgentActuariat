"""
TOOL CONTRACT — graphs.report_figures
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : graphs.report_figures
domain        : mortalite
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-01

DESCRIPTION
-----------
Génère les graphiques standards pour un rapport de mortalité
d'expérience. Chaque type de graphique correspond à un angle
d'analyse distinct du rapport (exposition, taux bruts, IC,
benchmarking, etc.). Les figures sont sauvegardées sur disque
et leur chemin est retourné pour intégration dans le PDF.

WHEN TO USE
-----------
Depuis le ReportAgent, après avoir lu les signaux du data_store
(Étapes A et B du raisonnement interprétatif). Choisir les types
de graphiques en fonction des anomalies détectées :
- exposition         : toujours inclure (vue générale du portefeuille)
- taux_log           : toujours inclure (taux bruts vs lissés)
- deces_ic95         : inclure si IC larges ou exposition faible
- abattements        : inclure si abatement_table disponible
- smr_par_decile     : inclure si hétérogénéité détectée dans les déciles
- logits             : inclure si lissage Whittaker utilisé
- comparaison_precedente : inclure si une table précédente est disponible

WHEN NOT TO USE
---------------
Ne pas appeler avant que smoothed_table soit disponible dans le data_store.
Ne pas appeler avec un type non listé — retourner erreur.

PREREQUISITES
-------------
required_tools: []
required_data_store_keys:
  - smoothed_table (pour taux_log, logits, comparaison_precedente)
  - exposure_table (pour exposition, deces_ic95)
  - benchmarking (pour abattements, smr_par_decile)
  - validation (pour deces_ic95)

INPUTS
------
params:
  figure_type:
    type    : string
    note    : >
      Type de graphique à générer. Valeurs acceptées :
        "exposition"             — effectifs et décès par âge
        "taux_log"               — taux bruts vs lissés en échelle log
        "deces_ic95"             — décès observés avec IC 95%
        "abattements"            — table d'abattements vs référence
        "smr_par_decile"         — SMR par décile d'âge (barres)
        "logits"                 — logits empiriques et courbe lissée
        "comparaison_precedente" — comparaison table courante / précédente
  output_path:
    type    : string
    default : "/tmp/figure_{figure_type}.png"
    note    : Chemin de sauvegarde du PNG. Créé automatiquement si absent.
  title:
    type    : string
    default : ""
    note    : Titre optionnel du graphique.
  sexe:
    type    : string
    default : ""
    note    : "H", "F" ou "" — ajouté au titre si précisé.

OUTPUTS
-------
data_store_keys_written: []
return_payload:
  figure_path : string — chemin absolu du PNG généré
  figure_type : string — type de graphique généré
  warning     : string | null

QUALITY GATES
-------------
BLOCKING:
  - figure_type non reconnu → erreur "Type de graphique inconnu : {type}"
  - données requises absentes du data_store → erreur descriptive
NON-BLOCKING:
  - aucune donnée dans la plage d'âges → warning + figure vide générée

ERROR HANDLING
--------------
error: "Type de graphique inconnu"
  → cause  : figure_type non dans la liste acceptée
  → action : retourner erreur, lister les types disponibles

error: "Données manquantes"
  → cause  : clé data_store absente
  → action : retourner erreur avec indication de la clé manquante

AGENT GUIDANCE
--------------
reasoning_hint: >
  Générer exposition et taux_log en priorité — ils figurent dans tous
  les rapports. Puis choisir les autres selon les signaux détectés.
  Le paramètre output_path doit pointer vers /tmp/ pour les rapports
  temporaires ou vers un dossier de sortie défini par le client.
exemplar_query: >
  non applicable

CATALOGUE METADATA
------------------
display_name      : Graphiques rapport de mortalité
short_description : Génère les 7 types de graphiques standards pour un rapport de mortalité.
domain            : mortalite
capability_group  : graphs
depends_on        [smoothed_table, exposure_table, benchmarking, validation]
required_by       : [build_pdf.certification_report, build_pdf.descriptive_report]
client_visible    : false
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_FIGURE_TYPES = {
    "exposition",
    "taux_log",
    "deces_ic95",
    "abattements",
    "smr_par_decile",
    "logits",
    "comparaison_precedente",
}

_REQUIRED_DATA: dict[str, list[str]] = {
    "exposition":             ["exposure_table"],
    "taux_log":               ["smoothed_table"],
    "deces_ic95":             ["exposure_table", "validation"],
    "abattements":            ["benchmarking"],
    "smr_par_decile":         ["benchmarking"],
    "logits":                 ["smoothed_table"],
    "comparaison_precedente": ["smoothed_table"],
}


# ─── entry point ──────────────────────────────────────────────────────────────

def run(data: dict, params: dict | None = None) -> dict:
    params      = params or {}
    figure_type = params.get("figure_type", "")
    output_path = params.get("output_path", f"/tmp/figure_{figure_type}.png")
    title       = params.get("title", "")
    sexe        = params.get("sexe", "")

    if not figure_type or figure_type not in _FIGURE_TYPES:
        return {
            "erreur": (
                f"Type de graphique inconnu : '{figure_type}'. "
                f"Types disponibles : {sorted(_FIGURE_TYPES)}"
            )
        }

    # Vérifier les données requises
    missing = [k for k in _REQUIRED_DATA[figure_type] if k not in data]
    if missing:
        return {
            "erreur": (
                f"Données manquantes pour '{figure_type}' : {missing}. "
                "Compléter le pipeline avant de générer ce graphique."
            )
        }

    # Construire le titre complet
    full_title = title
    if sexe and sexe not in full_title:
        full_title = f"{full_title} — {sexe}" if full_title else sexe

    # Créer le répertoire de sortie si nécessaire
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Dispatching
    warning = None
    try:
        if figure_type == "exposition":
            warning = _plot_exposition(data, out, full_title)
        elif figure_type == "taux_log":
            warning = _plot_taux_log(data, out, full_title)
        elif figure_type == "deces_ic95":
            warning = _plot_deces_ic95(data, out, full_title)
        elif figure_type == "abattements":
            warning = _plot_abattements(data, out, full_title)
        elif figure_type == "smr_par_decile":
            warning = _plot_smr_par_decile(data, out, full_title)
        elif figure_type == "logits":
            warning = _plot_logits(data, out, full_title)
        elif figure_type == "comparaison_precedente":
            warning = _plot_comparaison_precedente(data, out, full_title)
    except Exception as exc:
        return {
            "erreur":       f"Erreur lors de la génération du graphique : {exc}",
            "figure_type":  figure_type,
            "figure_path":  None,
            "warning":      None,
        }

    return {
        "figure_path": str(out.resolve()),
        "figure_type": figure_type,
        "warning":     warning,
    }


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rows_to_arrays(table: list[dict], *keys: str):
    """Extrait plusieurs colonnes d'une liste de dicts, retourne des arrays numpy."""
    arrays = []
    for k in keys:
        arrays.append(np.array([float(r.get(k, np.nan)) for r in table]))
    return arrays


def _style_axis(ax: plt.Axes, title: str = "") -> None:
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── plot functions ───────────────────────────────────────────────────────────

def _plot_exposition(data: dict, out: Path, title: str) -> str | None:
    table = data["exposure_table"]
    if not table:
        _save(plt.figure(), out)
        return "exposure_table vide — graphique non généré."

    ages, expo, deces = _rows_to_arrays(table, "age", "exposition_centrale", "deces")

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = ax1.twinx()

    ax1.bar(ages, expo, color="#4C72B0", alpha=0.7, label="Exposition centrale")
    ax2.step(ages, deces, color="#C44E52", linewidth=1.5, label="Décès observés", where="mid")

    ax1.set_xlabel("Âge", fontsize=10)
    ax1.set_ylabel("Exposition centrale (années-assurés)", fontsize=9, color="#4C72B0")
    ax2.set_ylabel("Décès observés", fontsize=9, color="#C44E52")
    _style_axis(ax1, title or "Exposition et décès par âge")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    fig.tight_layout()
    _save(fig, out)
    return None


def _plot_taux_log(data: dict, out: Path, title: str) -> str | None:
    table = data["smoothed_table"]
    if not table:
        _save(plt.figure(), out)
        return "smoothed_table vide — graphique non généré."

    ages      = np.array([float(r.get("age", np.nan)) for r in table])
    qx_brut   = np.array([float(r.get("qx_brut", np.nan)) for r in table])
    qx_lisse  = np.array([float(r.get("qx_lisse", r.get("qx_smooth", np.nan))) for r in table])
    qx_ref    = np.array([float(r.get("qx_reference", np.nan)) for r in table])

    fig, ax = plt.subplots(figsize=(9, 4.5))

    mask_b = qx_brut > 0
    mask_l = qx_lisse > 0
    mask_r = qx_ref > 0

    ax.scatter(ages[mask_b], np.log(qx_brut[mask_b]),
               s=14, color="#4C72B0", alpha=0.6, label="Taux bruts", zorder=3)
    ax.plot(ages[mask_l], np.log(qx_lisse[mask_l]),
            color="#C44E52", linewidth=2, label="Taux lissés")
    if mask_r.any():
        ax.plot(ages[mask_r], np.log(qx_ref[mask_r]),
                color="#55A868", linewidth=1.5, linestyle="--", label="Table référence")

    ax.set_xlabel("Âge", fontsize=10)
    ax.set_ylabel("log(qx)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{np.exp(y):.4f}" if abs(y) < 15 else ""
    ))
    _style_axis(ax, title or "Taux de mortalité (échelle log)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, out)
    return None


def _plot_deces_ic95(data: dict, out: Path, title: str) -> str | None:
    expo_table = data["exposure_table"]
    val_table  = data.get("validation", [])

    if not expo_table:
        _save(plt.figure(), out)
        return "exposure_table vide — graphique non généré."

    ages  = np.array([float(r.get("age", np.nan)) for r in expo_table])
    deces = np.array([float(r.get("deces", np.nan)) for r in expo_table])

    # Intervalles de confiance depuis validation si disponibles
    ic_lo = np.array([float(r.get("ic_lower", np.nan)) for r in val_table]) if val_table else None
    ic_hi = np.array([float(r.get("ic_upper", np.nan)) for r in val_table]) if val_table else None

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.bar(ages, deces, color="#4C72B0", alpha=0.75, label="Décès observés", zorder=2)

    if ic_lo is not None and ic_hi is not None and len(ic_lo) == len(ages):
        val_ages = np.array([float(r.get("age", np.nan)) for r in val_table])
        ax.errorbar(
            val_ages, (ic_lo + ic_hi) / 2,
            yerr=[(ic_hi - ic_lo) / 2, (ic_hi - ic_lo) / 2],
            fmt="none", ecolor="#C44E52", elinewidth=1, capsize=3,
            label="IC 95%", alpha=0.7, zorder=3,
        )

    ax.set_xlabel("Âge", fontsize=10)
    ax.set_ylabel("Nombre de décès", fontsize=10)
    _style_axis(ax, title or "Décès observés avec intervalles de confiance 95%")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, out)
    return None


def _plot_abattements(data: dict, out: Path, title: str) -> str | None:
    bm = data["benchmarking"]
    abat = bm.get("abatement_table", []) if isinstance(bm, dict) else []

    if not abat:
        _save(plt.figure(), out)
        return "abatement_table absente dans benchmarking — graphique non généré."

    ages   = np.array([float(r.get("age", np.nan)) for r in abat])
    abat_v = np.array([float(r.get("abattement", r.get("ratio", np.nan))) for r in abat])

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(ages, abat_v, color=np.where(abat_v < 1, "#55A868", "#C44E52"), alpha=0.8)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="Référence (1.0)")

    ax.set_xlabel("Âge", fontsize=10)
    ax.set_ylabel("Abattement (qx expérience / qx référence)", fontsize=9)
    ax.set_ylim(bottom=0)
    _style_axis(ax, title or "Abattements par âge vs table de référence")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, out)
    return None


def _plot_smr_par_decile(data: dict, out: Path, title: str) -> str | None:
    bm = data["benchmarking"]
    smr_deciles = bm.get("smr_par_decile", []) if isinstance(bm, dict) else []

    if not smr_deciles:
        _save(plt.figure(), out)
        return "smr_par_decile absent dans benchmarking — graphique non généré."

    labels = [str(r.get("decile", r.get("groupe", i))) for i, r in enumerate(smr_deciles)]
    smrs   = np.array([float(r.get("smr", np.nan)) for r in smr_deciles])
    colors = ["#55A868" if s < 1 else "#C44E52" for s in smrs]

    fig, ax = plt.subplots(figsize=(9, 4))

    x = np.arange(len(labels))
    ax.bar(x, smrs, color=colors, alpha=0.85)
    ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="SMR = 1.0")

    # Étiquettes de valeur
    for xi, s in zip(x, smrs):
        if not np.isnan(s):
            ax.text(xi, s + 0.01, f"{s:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("SMR", fontsize=10)
    _style_axis(ax, title or "SMR par décile d'âge")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, out)
    return None


def _plot_logits(data: dict, out: Path, title: str) -> str | None:
    table = data["smoothed_table"]
    if not table:
        _save(plt.figure(), out)
        return "smoothed_table vide — graphique non généré."

    ages     = np.array([float(r.get("age", np.nan)) for r in table])
    qx_brut  = np.array([float(r.get("qx_brut", np.nan)) for r in table])
    qx_lisse = np.array([float(r.get("qx_lisse", r.get("qx_smooth", np.nan))) for r in table])

    def logit(q):
        q = np.clip(q, 1e-8, 1 - 1e-8)
        return np.log(q / (1 - q))

    fig, ax = plt.subplots(figsize=(9, 4.5))

    mask_b = qx_brut > 0
    mask_l = qx_lisse > 0

    ax.scatter(ages[mask_b], logit(qx_brut[mask_b]),
               s=14, color="#4C72B0", alpha=0.6, label="Logits empiriques", zorder=3)
    ax.plot(ages[mask_l], logit(qx_lisse[mask_l]),
            color="#C44E52", linewidth=2, label="Logits lissés")

    ax.set_xlabel("Âge", fontsize=10)
    ax.set_ylabel("logit(qx)", fontsize=10)
    _style_axis(ax, title or "Logits empiriques et courbe lissée")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, out)
    return None


def _plot_comparaison_precedente(data: dict, out: Path, title: str) -> str | None:
    table   = data["smoothed_table"]
    prec    = data.get("precedent_table", [])

    if not table:
        _save(plt.figure(), out)
        return "smoothed_table vide — graphique non généré."

    ages     = np.array([float(r.get("age", np.nan)) for r in table])
    qx_curr  = np.array([float(r.get("qx_lisse", r.get("qx_smooth", np.nan))) for r in table])

    fig, ax = plt.subplots(figsize=(9, 4.5))

    mask = qx_curr > 0
    ax.plot(ages[mask], np.log(qx_curr[mask]),
            color="#4C72B0", linewidth=2, label="Table courante")

    if prec:
        ages_p  = np.array([float(r.get("age", np.nan)) for r in prec])
        qx_p    = np.array([float(r.get("qx_lisse", r.get("qx", np.nan))) for r in prec])
        mask_p  = qx_p > 0
        ax.plot(ages_p[mask_p], np.log(qx_p[mask_p]),
                color="#C44E52", linewidth=1.5, linestyle="--", label="Table précédente")

    ax.set_xlabel("Âge", fontsize=10)
    ax.set_ylabel("log(qx)", fontsize=10)
    _style_axis(ax, title or "Comparaison table courante / table précédente")
    ax.legend(fontsize=8)

    warning = None
    if not prec:
        warning = "precedent_table absent du data_store — comparaison partielle (table courante seulement)."

    fig.tight_layout()
    _save(fig, out)
    return warning
