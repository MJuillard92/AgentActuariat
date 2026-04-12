"""
TOOL CONTRACT — statistical_analysis.portfolio_summary
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : statistical_analysis.portfolio_summary
domain        : descriptive
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Calcule le résumé global du portefeuille : volume (contrats, décès), période
couverte, exposition totale en personne-années, statistiques d'âge, répartition
par sexe, et indicateurs de qualité des données. Premier outil à appeler dans
toute analyse descriptive.

WHEN TO USE
-----------
Appeler en premier dès qu'une analyse descriptive est demandée. Fournit le
contexte fondamental pour toutes les analyses suivantes. Obligatoire avant
build_pdf.descriptive_report.

WHEN NOT TO USE
---------------
Ne pas appeler si seule une construction de table de mortalité est demandée
(builder.exposure suffit). Ne pas relancer à l'identique si les colonnes
requises sont absentes.

PREREQUISITES
-------------
required_tools: []
required_data_store_keys: []
Note: reçoit df (DataFrame) directement.

INPUTS
------
params:
  observation_end:
    type    : string
    values  : date au format YYYY-MM-DD
    default : null
    note    : Date de fin d'observation pour tronquer les dates futures.
              Demander au client si des dates 2999 ou similaires sont détectées.
  max_reasonable_exit_date:
    type    : string
    values  : date au format YYYY-MM-DD
    default : 2100-01-01
    note    : Date maximale acceptable si observation_end est absent.

OUTPUTS
-------
data_store_keys_written:
  - summary : dict — résultat complet
return_payload:
  nb_contrats           : int
  nb_deces              : int
  exposition_totale_pa  : float
  taux_brut_deces_pour_1000_pa : float
  qualite_donnees       : dict
  warnings              : list[str]
  [et autres indicateurs — voir documentation complète]

QUALITY GATES
-------------
BLOCKING:
  - Aucune colonne date_entree détectée → résumé partiel. Clarifier avec client.
NON-BLOCKING:
  - warnings non vide → afficher chaque warning au client (dates invalides,
    durées négatives, etc.).

ERROR HANDLING
--------------
error: [aucun retour erreur structuré — outil robuste, s'adapte aux colonnes disponibles]
  → cause  : Colonnes absentes → résultats partiels (None pour les indicateurs non calculables).
  → action : Utiliser les indicateurs disponibles et signaler les limitations au client.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Toujours appeler en premier dans une analyse descriptive. Utiliser les
  warnings retournés pour informer le client des anomalies de données détectées.
  Si qualite_donnees contient des anomalies, proposer d'appeler
  statistical_analysis.data_quality pour les détails.
exemplar_query: >
  Que faire si portfolio_summary retourne un taux_brut_deces nul sur un grand portefeuille ?

CATALOGUE METADATA
------------------
display_name      : Résumé global du portefeuille
short_description : Calcule les indicateurs clés du portefeuille (volume, exposition, décès, âges).
domain            : descriptive
capability_group  : descriptive
depends_on        : []
required_by       : [build_pdf.descriptive_report]
client_visible    : true
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from agents.mortality.dictionary.column_schema import find_col as _find_col, COLUMN_SCHEMA as _CS


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_to_datetime(s: pd.Series) -> pd.Series:
    """Parse une série en datetime de façon robuste."""
    return pd.to_datetime(s, format="mixed", dayfirst=True, errors="coerce")


def _normalize_text(s: pd.Series) -> pd.Series:
    """Normalise en minuscules/trim, y compris pour colonnes non texte."""
    return s.astype(str).str.strip().str.lower()


def _detect_death_mask(s: pd.Series) -> pd.Series:
    """
    Détecte les décès à partir de valeurs usuelles.
    Exemples reconnus : deces, décès, dcd, dead, d, 1, true, yes, mort...
    """
    normalized = _normalize_text(s)

    death_values = {
        "deces", "décès", "decede", "décédé", "decedee", "décédée",
        "dcd", "dc", "dead", "death", "mort", "d", "1", "true", "yes", "oui"
    }

    return normalized.isin(death_values)


def _clean_exit_dates(
    exit_dates: pd.Series,
    observation_end: pd.Timestamp | None = None,
    max_reasonable_date: str = "2100-01-01",
) -> tuple[pd.Series, dict]:
    """
    Nettoie les dates de sortie aberrantes / sentinelles.
    - si observation_end est fourni : remplace les dates > observation_end par observation_end
    - sinon : les dates > max_reasonable_date sont mises à NaT
    """
    qc = {
        "nb_exit_dates_future_replaced": 0,
        "nb_exit_dates_aberrant_nullified": 0,
    }

    cleaned = exit_dates.copy()
    max_reasonable_ts = pd.Timestamp(max_reasonable_date)

    if observation_end is not None:
        mask_future = cleaned.notna() & (cleaned > observation_end)
        qc["nb_exit_dates_future_replaced"] = int(mask_future.sum())
        cleaned.loc[mask_future] = observation_end
    else:
        mask_aberrant = cleaned.notna() & (cleaned > max_reasonable_ts)
        qc["nb_exit_dates_aberrant_nullified"] = int(mask_aberrant.sum())
        cleaned.loc[mask_aberrant] = pd.NaT

    return cleaned, qc


def _round_or_none(value, digits=2):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    """
    Calcule le résumé global du portefeuille.

    Paramètres optionnels :
      - observation_end: str | datetime
          Date de fin d'observation à utiliser pour tronquer les dates de sortie fictives/futures.
          Exemple : "2024-12-31"
      - max_reasonable_exit_date: str
          Date max acceptable si observation_end n'est pas fourni. Au-delà : date de sortie ignorée.
          Défaut : "2100-01-01"

    Colonnes recherchées (noms canoniques ou synonymes courants) :
      - cause_sortie / statut / status → détection des décès
      - date_entree / ctreffet / entry_date → date d'entrée
      - date_sortie / exit_date → date de sortie
      - date_naissance / clinaiss / dob → date de naissance
      - age_entree → âge à l'entrée
      - age_sortie → âge à la sortie
      - duree_obs_ans → durée en personne-années
      - sexe / sexeref / gender → sexe

    Retourne un dict avec :
      - indicateurs de volume
      - indicateurs décès
      - période couverte
      - exposition totale
      - stats d'âge
      - répartition par sexe
      - qualité des données
      - colonnes disponibles
    """
    params = params or {}
    result: dict = {}

    # -------------------------------------------------------------------------
    # Paramètres
    # -------------------------------------------------------------------------
    observation_end = params.get("observation_end")
    if observation_end is not None:
        observation_end = pd.to_datetime(observation_end)

    max_reasonable_exit_date = params.get("max_reasonable_exit_date", "2100-01-01")

    # -------------------------------------------------------------------------
    # Colonnes
    # -------------------------------------------------------------------------
    death_col = _find_col(df, _CS["cause_sortie"]["candidates"])
    entry_col = _find_col(df, _CS["date_entree"]["candidates"])
    exit_col  = _find_col(df, _CS["date_sortie"]["candidates"])
    dob_col   = _find_col(df, _CS["date_naissance"]["candidates"])
    duree_col = _find_col(df, _CS["duree_obs_ans"]["candidates"])
    sexe_col  = _find_col(df, _CS["sexe"]["candidates"])

    # -------------------------------------------------------------------------
    # Base
    # -------------------------------------------------------------------------
    n = len(df)
    result["nb_contrats"] = int(n)
    result["nb_lignes_total"] = int(n)
    result["colonnes"] = list(df.columns)

    # -------------------------------------------------------------------------
    # Qualité des données
    # -------------------------------------------------------------------------
    quality = {
        "nb_dates_entree_invalides": None,
        "nb_dates_sortie_invalides": None,
        "nb_dates_naissance_invalides": None,
        "nb_durees_negatives": None,
        "nb_ages_negatifs": None,
        "nb_sorties_avant_entrees": None,
    }

    # -------------------------------------------------------------------------
    # Décès
    # -------------------------------------------------------------------------
    is_dead = None
    if death_col:
        is_dead = _detect_death_mask(df[death_col])
        nb_deces = int(is_dead.sum())
        result["nb_deces"] = nb_deces
        # proportion de lignes marquées en décès, pas un vrai taux actuariel
        result["proportion_deces_lignes"] = _round_or_none(nb_deces / n, 6) if n > 0 else None
    else:
        result["nb_deces"] = None
        result["proportion_deces_lignes"] = None

    # -------------------------------------------------------------------------
    # Parsing des dates
    # -------------------------------------------------------------------------
    entry_dates = _safe_to_datetime(df[entry_col]) if entry_col else None
    exit_dates = _safe_to_datetime(df[exit_col]) if exit_col else None
    dob_dates = _safe_to_datetime(df[dob_col]) if dob_col else None

    if entry_col:
        quality["nb_dates_entree_invalides"] = int(entry_dates.isna().sum())
    if exit_col:
        quality["nb_dates_sortie_invalides"] = int(exit_dates.isna().sum())
    if dob_col:
        quality["nb_dates_naissance_invalides"] = int(dob_dates.isna().sum())

    # Nettoyage des dates de sortie
    if exit_dates is not None:
        exit_dates, exit_qc = _clean_exit_dates(
            exit_dates,
            observation_end=observation_end,
            max_reasonable_date=max_reasonable_exit_date,
        )
        quality.update(exit_qc)

    # -------------------------------------------------------------------------
    # Période couverte
    # -------------------------------------------------------------------------
    if entry_dates is not None:
        valid = entry_dates.dropna()
        if len(valid) > 0:
            result["date_entree_min"] = str(valid.min().date())
            result["date_entree_max"] = str(valid.max().date())

    if exit_dates is not None:
        valid = exit_dates.dropna()
        if len(valid) > 0:
            result["date_sortie_min"] = str(valid.min().date())
            result["date_sortie_max"] = str(valid.max().date())

    # -------------------------------------------------------------------------
    # Exposition totale
    # -------------------------------------------------------------------------
    exposition_totale_pa = None

    if duree_col and pd.api.types.is_numeric_dtype(df[duree_col]):
        durees = pd.to_numeric(df[duree_col], errors="coerce")
        quality["nb_durees_negatives"] = int((durees < 0).fillna(False).sum())

        durees = durees.clip(lower=0)
        exposition_totale_pa = float(durees.sum())
        result["source_exposition"] = "colonne_duree_obs_ans"

    elif entry_dates is not None and exit_dates is not None:
        raw_durees = (exit_dates - entry_dates).dt.days / 365.25
        quality["nb_sorties_avant_entrees"] = int((raw_durees < 0).fillna(False).sum())

        durees = raw_durees.clip(lower=0)
        exposition_totale_pa = float(durees.fillna(0).sum())
        quality["nb_durees_negatives"] = int((raw_durees < 0).fillna(False).sum())
        result["source_exposition"] = "calculee_depuis_dates"

    else:
        result["source_exposition"] = None

    result["exposition_totale_pa"] = _round_or_none(exposition_totale_pa, 2)
    result["exposition_moyenne_pa_par_contrat"] = (
        _round_or_none(exposition_totale_pa / n, 4) if exposition_totale_pa is not None and n > 0 else None
    )

    # Taux brut rapporté à l'exposition
    if result.get("nb_deces") is not None and exposition_totale_pa not in (None, 0):
        result["taux_brut_deces_par_pa"] = _round_or_none(result["nb_deces"] / exposition_totale_pa, 6)
        result["taux_brut_deces_pour_1000_pa"] = _round_or_none(1000 * result["nb_deces"] / exposition_totale_pa, 3)
    else:
        result["taux_brut_deces_par_pa"] = None
        result["taux_brut_deces_pour_1000_pa"] = None

    # -------------------------------------------------------------------------
    # Âges à l'entrée
    # -------------------------------------------------------------------------
    ages = None

    if dob_dates is not None and entry_dates is not None:
        ages = (entry_dates - dob_dates).dt.days / 365.25
        result["source_age"] = "calcule_depuis_date_naissance_et_entree"
    else:
        result["source_age"] = None

    if ages is not None:
        quality["nb_ages_negatifs"] = int((ages < 0).fillna(False).sum())
        ages = ages.clip(lower=0).dropna()

        if len(ages) > 0:
            result["age_min"] = _round_or_none(ages.min(), 1)
            result["age_max"] = _round_or_none(ages.max(), 1)
            result["age_moyen"] = _round_or_none(ages.mean(), 1)
            result["age_median"] = _round_or_none(ages.median(), 1)
            result["age_q25"] = _round_or_none(ages.quantile(0.25), 1)
            result["age_q75"] = _round_or_none(ages.quantile(0.75), 1)

    # -------------------------------------------------------------------------
    # Sexe
    # -------------------------------------------------------------------------
    if sexe_col:
        s = _normalize_text(df[sexe_col])

        mapping = {
            "m": "H",
            "h": "H",
            "male": "H",
            "homme": "H",
            "1": "H",
            "f": "F",
            "female": "F",
            "femme": "F",
            "2": "F",
        }

        s_norm = s.map(mapping).fillna("AUTRE")
        counts = s_norm.value_counts(dropna=False).to_dict()
        result["repartition_sexe"] = {str(k): int(v) for k, v in counts.items()}
        result["repartition_sexe_pct"] = {
            str(k): _round_or_none(v / n, 4) if n > 0 else None
            for k, v in counts.items()
        }

    # -------------------------------------------------------------------------
    # Warnings qualité
    # -------------------------------------------------------------------------
    warnings = []

    if quality.get("nb_sorties_avant_entrees", 0) not in (None, 0):
        warnings.append("Certaines dates de sortie sont antérieures aux dates d'entrée.")

    if quality.get("nb_exit_dates_future_replaced", 0) not in (None, 0):
        warnings.append("Certaines dates de sortie futures/sentinelles ont été tronquées à observation_end.")

    if quality.get("nb_exit_dates_aberrant_nullified", 0) not in (None, 0):
        warnings.append("Certaines dates de sortie aberrantes ont été ignorées.")

    if result.get("exposition_totale_pa") in (None, 0) and n > 0:
        warnings.append("Exposition totale nulle ou indisponible : vérifier les dates ou la durée d'observation.")

    if result.get("nb_deces", 0) and result.get("taux_brut_deces_par_pa") is None:
        warnings.append("Des décès ont été détectés mais le taux actuariel n'a pas pu être calculé faute d'exposition.")

    result["qualite_donnees"] = quality
    result["warnings"] = warnings

    return result