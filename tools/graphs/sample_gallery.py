"""
TOOL CONTRACT — graphs.sample_gallery
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : graphs.sample_gallery
domain        : descriptive
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Génère une galerie de mini-graphiques avec données synthétiques pour montrer
au client les types de rendus disponibles avant toute analyse. Aucune donnée
réelle n'est requise. Utile pour orienter le client sur ce que le système
peut produire.

WHEN TO USE
-----------
Appeler en début de session si le client demande "quels graphiques tu peux
faire", "montre-moi des exemples", "qu'est-ce que tu peux produire" ou
formulation similaire. Appeler AVANT tout autre outil si la demande porte
sur les capacités visuelles du système.

WHEN NOT TO USE
---------------
Ne pas appeler si le client a déjà vu la galerie dans la session.
Ne pas appeler si la demande est une analyse concrète (et non une demande
de visualisation des capacités).

PREREQUISITES
-------------
required_tools: []
required_data_store_keys: []

INPUTS
------
params:
  filter:
    type    : string
    values  : descriptive | builder | all
    default : all
    note    : "descriptive" = pyramide, séries, segmentation. "builder" = exposition,
              taux bruts/lissés, SMR. "all" = les deux catégories.

OUTPUTS
-------
data_store_keys_written: []
return_payload:
  samples   : list[dict] — [{chart, title, description, category, image_b64}]
  n_samples : int — nombre de graphiques générés

QUALITY GATES
-------------
BLOCKING: []
NON-BLOCKING:
  - erreur dans un graphique individuel → l'entrée contiendra une clé "erreur"
    mais les autres graphiques sont toujours retournés.

ERROR HANDLING
--------------
error: [aucun retour erreur global — erreurs individuelles dans chaque sample]
  → cause  : Exception lors de la génération d'un graphique synthétique.
  → action : Afficher les autres graphiques disponibles. Logger l'erreur.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Utiliser filter="descriptive" pour une demande d'analyse descriptive,
  filter="builder" pour une demande de construction de table, filter="all"
  si la demande est générale ou ambiguë.
  Après la galerie, demander : "Parmi ces rendus, lesquels souhaitez-vous
  inclure dans votre étude ?"
exemplar_query: >
  Quels rendus visuels proposer pour une étude de mortalité d'expérience ?

CATALOGUE METADATA
------------------
display_name      : Galerie de rendus disponibles
short_description : Montre des exemples de graphiques disponibles avec données synthétiques.
domain            : descriptive
capability_group  : graphs
depends_on        : []
required_by       : []
client_visible    : true
"""
from __future__ import annotations

import base64
import io
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

_BG    = "#FBF8F1"
_GRID  = "#E8E3D8"
_BLUE  = "#2C5F8A"
_RED   = "#C0392B"
_GREEN = "#27AE60"
_ORANGE = "#E67E22"


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _theme(fig, axes):
    fig.patch.set_facecolor(_BG)
    for ax in (axes if isinstance(axes, list) else [axes]):
        ax.set_facecolor(_BG)
        ax.grid(True, color=_GRID, linewidth=0.7, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


# ── Graphiques descriptifs ────────────────────────────────────────────────────

def _sample_age_pyramid() -> str:
    bands = ["20-29","30-39","40-49","50-59","60-69","70-79","80+"]
    rng = np.random.default_rng(42)
    peak = np.array([0.4, 0.8, 1.0, 1.0, 0.85, 0.55, 0.25])
    h = (rng.integers(200, 700, len(bands)) * peak).astype(int)
    f = (rng.integers(200, 680, len(bands)) * peak).astype(int)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(bands))
    ax.bar(x - 0.2, h, 0.38, color=_BLUE, alpha=0.82, label="Hommes")
    ax.bar(x + 0.2, f, 0.38, color=_RED,  alpha=0.82, label="Femmes")
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=7)
    ax.set_ylabel("Contrats", fontsize=7)
    ax.set_title("Pyramide des âges", fontsize=9, loc="left", fontweight="bold")
    ax.legend(fontsize=7, facecolor=_BG)
    _theme(fig, [ax])
    fig.tight_layout()
    return _to_b64(fig)


def _sample_time_series() -> str:
    years = np.arange(2010, 2024)
    rng = np.random.default_rng(42)
    exposition = np.cumsum(rng.integers(-200, 300, len(years))) + 6000
    deces = (exposition * 0.009 + rng.integers(-5, 12, len(years))).astype(int)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
    ax1.fill_between(years, exposition, alpha=0.3, color=_BLUE)
    ax1.plot(years, exposition, color=_BLUE, lw=1.8)
    ax1.set_ylabel("Exposition (P-A)", fontsize=7)
    ax1.set_title("Exposition & décès par année", fontsize=9, loc="left", fontweight="bold")
    ax2.bar(years, deces, color=_RED, alpha=0.8, edgecolor="none")
    ax2.set_ylabel("Décès", fontsize=7)
    ax2.set_xlabel("Année", fontsize=7)
    _theme(fig, [ax1, ax2])
    fig.tight_layout()
    return _to_b64(fig)


def _sample_segmentation() -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.5))
    colors = [_BLUE, _RED, _GREEN, _ORANGE]
    ax1.pie([52, 48], labels=["Hommes", "Femmes"], autopct="%1.0f%%",
            colors=[_BLUE, _RED], startangle=90)
    ax1.set_title("Sexe", fontsize=8)
    ax2.pie([38, 29, 21, 12], labels=["Prod A","Prod B","Prod C","Autre"],
            autopct="%1.0f%%", colors=colors, startangle=90)
    ax2.set_title("Produit", fontsize=8)
    fig.suptitle("Répartitions du portefeuille", fontsize=9, fontweight="bold")
    fig.patch.set_facecolor(_BG)
    fig.tight_layout()
    return _to_b64(fig)


# ── Graphiques builder ────────────────────────────────────────────────────────

def _sample_exposure_by_age() -> str:
    ages = np.arange(30, 91)
    rng = np.random.default_rng(42)
    expo = np.exp(-0.04 * (ages - 55)**2 / 100) * 1200 + rng.integers(0, 80, len(ages))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(ages, expo, color=_BLUE, alpha=0.8, edgecolor="none", width=0.9)
    ax.set_xlabel("Âge", fontsize=7)
    ax.set_ylabel("Exposition E_x (P-A)", fontsize=7)
    ax.set_title("Exposition centrale par âge", fontsize=9, loc="left", fontweight="bold")
    _theme(fig, [ax])
    fig.tight_layout()
    return _to_b64(fig)


def _sample_crude_smoothed() -> str:
    ages = np.arange(30, 91)
    rng = np.random.default_rng(42)
    qx_true = 0.00004 * np.exp(0.088 * (ages - 30))
    noise = 1 + rng.normal(0, 0.18, len(ages))
    qx_crude = np.clip(qx_true * noise, 1e-5, 0.5)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(ages, qx_crude, s=14, color=_RED, alpha=0.55, label="Taux bruts q̂(x)", zorder=3)
    ax.plot(ages, qx_true, color=_BLUE, lw=2, label="Taux lissés — Whittaker", zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("Âge", fontsize=7)
    ax.set_ylabel("q(x)", fontsize=7)
    ax.set_title("Taux bruts vs lissés (log)", fontsize=9, loc="left", fontweight="bold")
    ax.legend(fontsize=7, facecolor=_BG)
    _theme(fig, [ax])
    fig.tight_layout()
    return _to_b64(fig)


def _sample_smr() -> str:
    decades = ["30-39","40-49","50-59","60-69","70-79","80-89"]
    rng = np.random.default_rng(42)
    smr = 0.72 + rng.random(len(decades)) * 0.65

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = [_GREEN if s < 1.0 else _ORANGE for s in smr]
    bars = ax.bar(decades, smr * 100, color=colors, alpha=0.85, edgecolor="none")
    ax.axhline(100, color="#333", lw=1.4, linestyle="--", label="Référence TH0002")
    ax.set_ylabel("SMR (%)", fontsize=7)
    ax.set_title("SMR par décennie vs TH0002", fontsize=9, loc="left", fontweight="bold")
    ax.set_ylim(0, 170)
    for bar, v in zip(bars, smr * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2,
                f"{v:.0f}%", ha="center", fontsize=7)
    ax.legend(fontsize=7, facecolor=_BG)
    _theme(fig, [ax])
    fig.tight_layout()
    return _to_b64(fig)


# ── Catalogue ─────────────────────────────────────────────────────────────────

_SAMPLES = [
    ("age_pyramid",    "Pyramide des âges",             "Distribution des assurés par tranche d'âge et sexe.",              "descriptive", _sample_age_pyramid),
    ("time_series",    "Exposition & décès par année",  "Évolution annuelle de l'exposition (P-A) et des décès observés.",  "descriptive", _sample_time_series),
    ("segmentation",   "Répartitions (camemberts)",     "Ventilation du portefeuille par sexe, produit, statut.",           "descriptive", _sample_segmentation),
    ("exposure",       "Exposition par âge",            "E_x (personne-années) par âge : volume de données disponible.",    "builder",     _sample_exposure_by_age),
    ("crude_smoothed", "Taux bruts vs lissés (log)",    "q(x) bruts (points) et lissés Whittaker (courbe) en log.",         "builder",     _sample_crude_smoothed),
    ("smr",            "SMR par décennie",              "Ratio mortalité observée / table de référence TH0002 ou TF0002.",   "builder",     _sample_smr),
]


def run(data: dict | None, params: dict | None = None) -> dict:
    data   = data   or {}
    params = params or {}
    filter_type = params.get("filter", "all")

    samples = []
    for chart, title, description, category, gen_fn in _SAMPLES:
        if filter_type != "all" and category != filter_type:
            continue
        try:
            img = gen_fn()
            samples.append({
                "chart":       chart,
                "title":       title,
                "description": description,
                "category":    category,
                "image_b64":   img,
            })
        except Exception as exc:
            samples.append({
                "chart":       chart,
                "title":       title,
                "description": description,
                "category":    category,
                "image_b64":   None,
                "erreur":      str(exc),
            })

    return {"samples": samples, "n_samples": len(samples)}
