"""
TOOL CONTRACT — statistical_analysis.time_series
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : statistical_analysis.time_series
domain        : descriptive
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Construit une série temporelle annuelle du portefeuille : nombre d'entrées,
de décès, et exposition en personne-années par année calendaire. Détecte
automatiquement les anomalies temporelles (années à exposition anormalement
faible). Résultat utilisé par graphs.analysis_plots (chart=time_series).

WHEN TO USE
-----------
Appeler dans toute analyse descriptive pour visualiser l'évolution temporelle
du portefeuille. Utile pour détecter des ruptures de collecte ou des anomalies
de volume.

WHEN NOT TO USE
---------------
Ne pas appeler si date_entree est absente (retourne erreur).

PREREQUISITES
-------------
required_tools: []
required_data_store_keys: []
Note: reçoit df (DataFrame) directement.

INPUTS
------
params: {}
  Note: aucun paramètre requis.

OUTPUTS
-------
data_store_keys_written:
  - series : dict — résultat complet (serie, annee_min, annee_max, nb_annees, anomalies)
return_payload:
  serie     : list[dict] — annee, nb_entres, nb_deces, exposition_pa par année
  annee_min : int
  annee_max : int
  nb_annees : int
  anomalies : list[str] — années avec données manquantes ou exposition faible

QUALITY GATES
-------------
BLOCKING:
  - Colonne date_entree absente → retourne erreur.
NON-BLOCKING:
  - anomalies non vide → signaler les années problématiques au client.
    Peut indiquer des trous dans la collecte de données.

ERROR HANDLING
--------------
error: "Colonne date d'entrée introuvable. Colonnes : [...]"
  → cause  : date_entree absent du DataFrame.
  → action : Vérifier le dictionnaire de données avec le client.
error: "Aucune date d'entrée valide."
  → cause  : Toutes les dates d'entrée sont invalides ou manquantes.
  → action : Appeler statistical_analysis.data_quality pour diagnostiquer.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Si anomalies est non vide, expliquer au client les années concernées.
  Une exposition anormalement faible peut indiquer un démarrage de contrat
  en milieu d'année ou une fin de collecte. Le résultat "series" est requis
  pour graphs.analysis_plots (chart=time_series).
exemplar_query: >
  Comment interpréter une série temporelle avec une chute soudaine d'exposition ?

CATALOGUE METADATA
------------------
display_name      : Évolution temporelle
short_description : Calcule l'évolution annuelle de l'exposition et des décès par année.
domain            : descriptive
capability_group  : descriptive
depends_on        : []
required_by       : [graphs.analysis_plots, build_pdf.descriptive_report]
client_visible    : true
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from agents.mortality.dictionary.column_schema import find_col as _find_col, COLUMN_SCHEMA as _CS


def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    """
    Construit une série temporelle annuelle :
      - nb_contrats_entres : contrats dont la date d'entrée est dans l'année
      - nb_deces           : décès survenus dans l'année
      - exposition_pa      : personne-années d'exposition dans l'année (approx.)

    Détecte aussi les années avec données manquantes ou volumes anormalement bas.
    """
    entry_col  = _find_col(df, _CS["date_entree"]["candidates"])
    exit_col   = _find_col(df, _CS["date_sortie"]["candidates"])
    death_col  = _find_col(df, _CS["cause_sortie"]["candidates"])

    if not entry_col:
        return {"erreur": "Colonne date d'entrée introuvable. Colonnes : " + str(list(df.columns))}

    # Parsing des dates
    df = df.copy()
    df["_entree"] = pd.to_datetime(df[entry_col], format="mixed", dayfirst=True, errors="coerce")
    if exit_col:
        df["_sortie"] = pd.to_datetime(df[exit_col], format="mixed", dayfirst=True, errors="coerce")
    else:
        df["_sortie"] = pd.NaT

    if death_col:
        col = df[death_col].astype(str).str.lower().str.strip()
        df["_is_dead"] = col.isin(["deces", "décès", "decede", "décédé", "d", "1", "true", "mort", "dead", "dcd"])
    else:
        df["_is_dead"] = False

    valid = df.dropna(subset=["_entree"])
    if len(valid) == 0:
        return {"erreur": "Aucune date d'entrée valide."}

    year_min = int(valid["_entree"].dt.year.min())
    year_max = int(valid["_sortie"].dt.year.max()) if exit_col and valid["_sortie"].notna().any() \
               else int(valid["_entree"].dt.year.max())

    rows = []
    for year in range(year_min, year_max + 1):
        # Contrats actifs dans l'année (entrée avant fin d'année, sortie après début)
        start = pd.Timestamp(year, 1, 1)
        end   = pd.Timestamp(year, 12, 31)

        entres = valid[valid["_entree"].dt.year == year]
        nb_entres = len(entres)

        if exit_col:
            actifs = valid[
                (valid["_entree"] <= end) &
                (valid["_sortie"].isna() | (valid["_sortie"] >= start))
            ]
        else:
            actifs = valid[valid["_entree"].dt.year <= year]

        # Décès dans l'année
        if exit_col:
            nb_deces = int(valid[
                valid["_is_dead"] &
                (valid["_sortie"].dt.year == year)
            ].shape[0])
        else:
            nb_deces = 0

        # Exposition approx. (années de présence dans l'année)
        if exit_col and len(actifs) > 0:
            ent_clip = actifs["_entree"].clip(lower=start)
            sor_clip = actifs["_sortie"].fillna(end).clip(upper=end)
            expo = ((sor_clip - ent_clip).dt.days.clip(lower=0) / 365.25).sum()
        else:
            expo = len(actifs)  # approximation : 1 personne-année par contrat actif

        rows.append({
            "annee":        year,
            "nb_entres":    nb_entres,
            "nb_deces":     nb_deces,
            "exposition_pa": round(float(expo), 1),
        })

    series = pd.DataFrame(rows).set_index("annee")

    # Détection d'anomalies
    anomalies = []
    mean_expo = series["exposition_pa"].mean()
    for year, row in series.iterrows():
        if row["exposition_pa"] == 0 and row["nb_entres"] == 0:
            anomalies.append(f"{year} : aucune donnée")
        elif mean_expo > 0 and row["exposition_pa"] < 0.1 * mean_expo:
            anomalies.append(f"{year} : exposition anormalement faible ({row['exposition_pa']:.0f} PA)")

    result = {
        "serie": series.reset_index().to_dict(orient="records"),
        "annee_min": year_min,
        "annee_max": year_max,
        "nb_annees": year_max - year_min + 1,
    }
    if anomalies:
        result["anomalies"] = anomalies

    return result
