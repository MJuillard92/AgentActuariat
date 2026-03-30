"""
report_agent/test_report_agent.py
Test de bout-en-bout : payload synthétique → validation → PDF.

Lance avec :
    python -m report_agent.test_report_agent
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from report_agent.validate_payload import validate
from report_agent.generate_report import generate_mortality_report
from report_payload_builder import build_report_payload, build_exposure_deciles


# ─── Données synthétiques réalistes ───────────────────────────────────────────

def _build_test_payload() -> dict:
    rng = np.random.default_rng(42)
    ages = np.arange(25, 126, dtype=float)   # 101 âges
    n = len(ages)

    # Taux de référence Gompertz-Makeham
    A, B, c = 0.0003, 0.000025, 0.10
    q_ref = np.clip(A + B * np.exp(c * (ages - 50)), 5e-5, 0.9)

    # Exposition décroissante avec l'âge
    exposure = rng.integers(200, 3000, n).astype(float) * np.linspace(1.0, 0.2, n)
    exposure = np.maximum(exposure, 20.0)

    # Décès simulés
    deaths = rng.poisson(q_ref * exposure).astype(float)
    q_brut = np.where(exposure > 0, deaths / exposure, q_ref)

    # Lissage (noyau gaussien — approx Whittaker-Henderson)
    kernel = np.exp(-0.5 * (np.arange(-4, 5, dtype=float) / 2.0) ** 2)
    kernel /= kernel.sum()
    q_lisse = np.convolve(q_brut, kernel, mode="same")
    q_lisse = np.maximum(q_lisse, 1e-5)

    # IC 95 % (approximation Poisson)
    ic_inf = np.maximum(0.0, q_lisse - 1.96 * np.sqrt(q_lisse / np.maximum(exposure, 1.0)))
    ic_sup = q_lisse + 1.96 * np.sqrt(q_lisse / np.maximum(exposure, 1.0))

    # SMR global
    D_obs = float(deaths.sum())
    D_exp = float((q_ref * exposure).sum())
    smr = D_obs / D_exp
    D_i = D_obs
    lo = (D_i / D_exp) * (1 - 1 / (9 * D_i) - 1.96 / (3 * np.sqrt(D_i))) ** 3
    hi = ((D_i + 1) / D_exp) * (1 - 1 / (9 * (D_i + 1)) + 1.96 / (3 * np.sqrt(D_i + 1))) ** 3

    # χ² (approximation)
    chi2 = float(
        np.sum(
            (deaths - q_ref * exposure) ** 2 / np.maximum(q_ref * exposure, 1.0)
        )
    )
    ddl = n - 2
    try:
        from scipy import stats as _stats
        pval = float(1 - _stats.chi2.cdf(chi2, ddl))
    except ImportError:
        pval = 0.05

    abat_global = float(np.sum(q_lisse * exposure) / np.sum(q_ref * exposure))

    portfolio_info = {
        "n_assures": 50_000,
        "n_contrats_actifs": 42_000,
        "type_contrat": "vie_entiere",
        "periode_debut": "2010-01-01",
        "periode_fin": "2023-12-31",
        "age_min": 25,
        "age_max": 125,
        "segmentation": "global",
        "table_reference": "TH00-02",
    }

    qualite_info = {
        "traitements_appliques": [
            {"nom": "Nettoyage âges extrêmes", "description": "Âges <25 et >125 exclus"},
            {"nom": "Imputation exposition", "description": "Périodes incomplètes en début/fin"},
        ],
        "stats_annuelles": [
            {
                "annee": yr,
                "exposition": int(exposure.sum() / 14),
                "age_moyen": round(float(np.average(ages, weights=exposure)), 1),
                "deces": int(deaths.sum() / 14),
            }
            for yr in range(2010, 2024)
        ],
    }

    return build_report_payload(
        ages=ages,
        exposure=exposure,
        deaths_observed=deaths,
        q_brut=q_brut,
        q_lisse=q_lisse,
        ic_inf=ic_inf,
        ic_sup=ic_sup,
        q_ref=q_ref,
        methode="whittaker_henderson",
        parametres={"lambda": 1000, "ordre": 2},
        smr_global=smr,
        smr_ic_inf=lo,
        smr_ic_sup=hi,
        chi2_stat=chi2,
        chi2_ddl=ddl,
        chi2_pvalue=pval,
        abattement_global=abat_global,
        portfolio_info=portfolio_info,
        qualite_info=qualite_info,
        trace_info={"version": "1.0", "date": "2024-01-01"},
    )


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_validation_ok():
    payload = _build_test_payload()
    result = validate(payload)
    assert result.valid, f"Validation échouée :\n{result.refusal_message()}"
    print("  ✓ Validation OK")


def test_validation_fails_on_empty():
    result = validate({})
    assert not result.valid
    assert len(result.errors) > 0
    print("  ✓ Validation rejette un payload vide")


def test_pdf_generation():
    payload = _build_test_payload()
    out = os.path.join(tempfile.gettempdir(), "test_rapport_certification.pdf")
    path = generate_mortality_report(payload, output_path=out)
    size = os.path.getsize(path)
    assert size > 50_000, f"PDF trop petit : {size} octets"
    print(f"  ✓ PDF généré : {path} ({size // 1024} Ko)")
    return path


def test_deciles_are_exposure_based():
    """Vérifie que les déciles sont par quantiles d'exposition, pas par tranche fixe."""
    rng = np.random.default_rng(0)
    ages = np.arange(30, 81, dtype=float)
    exposure = rng.integers(100, 2000, len(ages)).astype(float)
    deaths = rng.integers(0, 10, len(ages)).astype(float)
    q_ref = np.full(len(ages), 0.01)

    deciles = build_exposure_deciles(ages, exposure, deaths, q_ref)
    assert len(deciles) >= 3
    # Chaque décile doit avoir environ 10 % de l'exposition totale
    total = sum(d["exposure"] for d in deciles)
    for d in deciles:
        frac = d["exposure"] / total
        assert 0.03 < frac < 0.35, f"Décile hors borne : {d['tranche_label']} = {frac:.2%}"
    print(f"  ✓ Déciles d'exposition OK ({len(deciles)} tranches)")


def test_fallback():
    """Vérifie que le fallback mécanique produit un narratif structurellement correct."""
    from report_agent.generate_report import generate_narratif_fallback
    payload = _build_test_payload()
    narratif = generate_narratif_fallback(payload)
    _check_narratif_structure(narratif)
    print("  ✓ Narratif fallback structurellement correct")


def test_narratif_structure():
    """Vérifie que le narratif (fallback) contient toutes les clés attendues par le renderer."""
    from report_agent.generate_report import generate_narratif_fallback
    payload = _build_test_payload()
    narratif = generate_narratif_fallback(payload)
    _check_narratif_structure(narratif)

    # Vérifications de contenu minimal
    assert len(narratif["preambule"]) > 50, "Préambule trop court"
    assert len(narratif["section_1_contrats"]["paragraphes"]) >= 1
    assert len(narratif["section_3_methodologie"]["commentaire_smr"]) > 10
    assert isinstance(narratif["section_5_commentaires"]["alertes"], list)
    print("  ✓ Structure et contenu du narratif OK")


def _check_narratif_structure(narratif: dict) -> None:
    """Vérifie la présence de toutes les clés du schéma narratif."""
    required_keys = [
        "preambule",
        "section_1_contrats",
        "section_2_donnees",
        "section_3_methodologie",
        "section_4_construction",
        "section_5_commentaires",
        "section_6_conclusion",
    ]
    for k in required_keys:
        assert k in narratif, f"Clé manquante dans le narratif : {k}"

    assert "paragraphes" in narratif["section_1_contrats"]
    assert "paragraphes_avant_tableaux" in narratif["section_2_donnees"]
    assert "paragraphes_apres_tableaux" in narratif["section_2_donnees"]
    for k in ["intro", "commentaire_lissage", "commentaire_smr",
              "commentaire_chi2", "commentaire_abattement", "commentaire_deciles"]:
        assert k in narratif["section_3_methodologie"], f"section_3 manque : {k}"
    for k in ["intro_taux_bruts", "commentaire_taux_lisses", "commentaire_figure_taux",
              "intro_abattement", "commentaire_figure_abattement"]:
        assert k in narratif["section_4_construction"], f"section_4 manque : {k}"
    assert "paragraphes" in narratif["section_5_commentaires"]
    assert "alertes" in narratif["section_5_commentaires"]
    for k in ["synthese", "recommandations", "validation"]:
        assert k in narratif["section_6_conclusion"], f"section_6 manque : {k}"


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  report_agent — Tests de bout-en-bout")
    print("=" * 60 + "\n")

    test_validation_fails_on_empty()
    test_validation_ok()
    test_deciles_are_exposure_based()
    test_fallback()
    test_narratif_structure()
    pdf_path = test_pdf_generation()

    print("\n" + "=" * 60)
    print("  TOUS LES TESTS PASSENT")
    print(f"  PDF de démonstration : {pdf_path}")
    print("=" * 60 + "\n")
