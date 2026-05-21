"""
Chantier dates 2026-05-21 — Golden test (filet de non-régression numérique).

Fige les sorties de `builder.exposure` et `builder.crude_rates` sur un
portefeuille de référence (`tests/fixtures/golden_portfolio.csv`) qui exerce :
dates FR `JJ/MM/AAAA`, jour>12 (`28/11/2015`), sentinelles `2999`/`9999` ET
`2050` (l'année que l'ancienne regex ratait), et une date invalide `0/0/0`.

Rôle : toute modification du code de dates qui ferait bouger un chiffre
échoue ici. Les valeurs figées sont les valeurs CORRECTES (sentinelles
clippées à observation_end).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.builder.exposure import run as exposure_run
from tools.builder.crude_rates import run as crude_rates_run

_FIXTURE = Path(__file__).parent / "fixtures" / "golden_portfolio.csv"


def _load() -> pd.DataFrame:
    return pd.read_csv(_FIXTURE, dtype=str)


# ── Exposition — valeurs figées ─────────────────────────────────────────────

def test_golden_exposure_frozen_values() -> None:
    """Sorties exactes de builder.exposure sur le portefeuille de référence."""
    res = exposure_run(_load(), {})
    assert res.get("erreur") is None, res.get("erreur")
    assert res["total_exposure"] == 199.91
    assert res["total_deaths"] == 12
    # La ligne date_naissance=0/0/0 est exclue (1 ligne).
    assert res["lignes_exclues"] == 1


def test_golden_exposure_no_sentinel_leak() -> None:
    """Invariant de plausibilité : aucune sentinelle futur-lointain ne fuit.

    Le portefeuille a 5 contrats actifs (sentinelles 2999×2, 2050×2, 9999×1).
    S'ils étaient comptés à leur année faciale, l'exposition exploserait
    (un contrat entré en 2008 « sorti » en 2050 = ~42 ans). Le clipping à
    observation_end (~2019) borne le total. Un total > 240 prouverait une
    fuite de sentinelle (notamment le trou historique sur 2050)."""
    res = exposure_run(_load(), {})
    assert res["total_exposure"] < 240.0, (
        f"exposition {res['total_exposure']} trop haute — sentinelle fuitée ?"
    )


# ── Taux bruts (méthode centrale) — valeurs figées ──────────────────────────

def test_golden_crude_rates_central_frozen() -> None:
    """Sorties exactes de builder.crude_rates (méthode centrale)."""
    exp = exposure_run(_load(), {})
    cr = crude_rates_run({"exposure_table": exp["exposure_table"]},
                         {"method": "central"})
    assert cr.get("erreur") is None, cr.get("erreur")
    qx = cr["qx_table"]
    assert len(qx) == 71
    # 12 décès, conservés intégralement par l'agrégation par âge.
    assert sum(int(r.get("D_x") or 0) for r in qx) == 12
