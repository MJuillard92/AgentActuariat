"""
report_agent/generate_report.py
Orchestrateur : validation → narration → rendu PDF.

Pipeline :
    payload JSON → validate() → narratif dict → render_pdf() → PDF

Si llm_client fourni → agent LLM rédige le narratif.
Sinon → fallback mécanique (f-strings structurés).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from report_agent.validate_payload import validate
from report_agent.renderer import render_pdf
from report_agent.brief import Brief, check_brief_satisfiability, check_payload_against_brief


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def generate_mortality_report(
    payload: dict,
    output_path: str = "rapport_certification.pdf",
    llm_client: Any | None = None,
) -> str:
    """Génère le PDF. Lève ValueError si payload invalide.

    Args:
        payload:     Dict conforme au data contract (validate_payload.py).
        output_path: Chemin du PDF de sortie.
        llm_client:  Client LLM implémentant .complete(system, user) → str.
                     Si None, utilise le fallback mécanique.

    Returns:
        Chemin absolu du PDF généré.
    """
    result = validate(payload)
    if not result.valid:
        raise ValueError(result.refusal_message())

    if llm_client is not None:
        narratif = generate_narratif_llm(payload, llm_client)
    else:
        narratif = generate_narratif_fallback(payload)

    return render_pdf(narratif, payload, output_path)


# ─── Narration LLM ────────────────────────────────────────────────────────────

def generate_narratif_llm(payload: dict, llm_client: Any) -> dict:
    """Génère le narratif via un agent LLM.

    Charge le system prompt depuis system_prompt.md.
    Construit un résumé compressé du payload (sans les vecteurs bruts).
    Appelle llm_client.complete() et parse le JSON retourné.
    """
    # Charger le system prompt
    sp_path = Path(__file__).parent / "system_prompt.md"
    if sp_path.exists():
        system_prompt = sp_path.read_text(encoding="utf-8")
    else:
        system_prompt = (
            "Tu es un actuaire senior. Produis un JSON structuré avec le narratif "
            "du rapport de certification de table de mortalité. "
            "Réponds UNIQUEMENT avec le JSON, sans backticks."
        )

    summary = _build_payload_summary(payload)
    user_message = (
        "Voici les données du rapport actuariel. "
        "Rédige le narratif complet en JSON structuré.\n\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
    )

    raw = llm_client.complete(system_prompt, user_message)
    return _parse_narratif_json(raw)


def _parse_narratif_json(raw: str) -> dict:
    """Parse le JSON du narratif, en gérant les éventuels blocs markdown."""
    text = raw.strip()
    # Retirer les blocs markdown (```json ... ```)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    # Fallback : chercher un objet JSON dans le texte
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Si tout échoue, retourner un narratif minimal
    return generate_narratif_fallback({
        "portfolio": {"n_assures": 0, "n_contrats_actifs": 0, "type_contrat": "",
                      "periode_debut": "", "periode_fin": "", "age_min": 0, "age_max": 0,
                      "segmentation": "", "table_reference": ""},
        "donnees_brutes": {"ages": [], "exposure": [], "deaths_observed": [], "q_brut": []},
        "lissage": {"methode": "whittaker_henderson", "parametres": {},
                    "q_lisse": [], "ic_inf": [], "ic_sup": [], "q_ref": []},
        "validation": {"smr_global": 1.0, "smr_ic_inf": 0.9, "smr_ic_sup": 1.1,
                       "chi2_stat": 0.0, "chi2_ddl": 1, "chi2_pvalue": 0.5,
                       "abattement_global": 1.0},
        "deciles": [], "abattement": {"ages": [], "q_construit": [], "q_reference": [], "alpha": []},
        "qualite_donnees": {"traitements_appliques": [], "stats_annuelles": []},
    })


# ─── Résumé payload pour contexte LLM ────────────────────────────────────────

def _build_payload_summary(payload: dict) -> dict:
    """Compresse le payload pour le contexte LLM — sans les vecteurs bruts."""
    pf = payload["portfolio"]
    db = payload["donnees_brutes"]
    lis = payload["lissage"]
    val = payload["validation"]
    dec = payload["deciles"]
    abat = payload["abattement"]
    qd = payload["qualite_donnees"]

    ages = np.array(db["ages"], dtype=float)
    exp = np.array(db["exposure"], dtype=float)
    deaths = np.array(db["deaths_observed"], dtype=float)
    q_brut = np.array(db["q_brut"], dtype=float)
    q_lisse = np.array(lis["q_lisse"], dtype=float)
    q_ref = np.array(lis["q_ref"], dtype=float)
    alpha = np.array(abat["alpha"], dtype=float)

    total_d = float(deaths.sum())
    total_e = float(exp.sum())
    taux_brut_global = total_d / total_e if total_e > 0 else 0.0
    age_moyen = float(np.average(ages, weights=exp)) if exp.sum() > 0 else float(ages.mean())

    # Échantillon ~20 âges représentatifs
    step = max(1, len(ages) // 19)
    sample_idx = list(range(0, len(ages), step))[:20]
    echantillon = [
        {
            "age": int(ages[i]),
            "exposure": round(float(exp[i]), 1),
            "deaths": int(deaths[i]),
            "q_brut_permille": round(float(q_brut[i]) * 1000, 2),
            "q_lisse_permille": round(float(q_lisse[i]) * 1000, 2),
            "q_ref_permille": round(float(q_ref[i]) * 1000, 2),
            "abattement": round(float(alpha[i]), 3),
        }
        for i in sample_idx
        if i < len(ages)
    ]

    def _find_zones(mask: np.ndarray, label: str) -> list[dict]:
        zones = []
        idx = np.where(mask)[0]
        if not len(idx):
            return zones
        groups: list[tuple[int, int]] = []
        start = prev = idx[0]
        for i in idx[1:]:
            if i > prev + 2:
                groups.append((start, prev))
                start = i
            prev = i
        groups.append((start, prev))
        for gs, ge in groups[:3]:
            zones.append({
                "age_debut": int(ages[gs]),
                "age_fin": int(ages[ge]),
                "abattement_moyen": round(float(alpha[gs:ge + 1].mean()), 3),
                "exposition_totale": round(float(exp[gs:ge + 1].sum()), 0),
                "deces_total": int(deaths[gs:ge + 1].sum()),
            })
        return zones

    zones_supra  = _find_zones((alpha > 1.1) & (exp >= 100), "surmortalite")
    zones_infra  = _find_zones((alpha < 0.7) & (exp >= 100), "sous_mortalite")
    zones_faible = []
    idx_f = np.where(exp < 100)[0]
    if len(idx_f):
        groups_f: list[tuple[int, int]] = []
        s, pv = idx_f[0], idx_f[0]
        for i in idx_f[1:]:
            if i > pv + 2:
                groups_f.append((s, pv))
                s = i
            pv = i
        groups_f.append((s, pv))
        for gs, ge in groups_f[:3]:
            zones_faible.append({
                "age_debut": int(ages[gs]),
                "age_fin": int(ages[ge]),
                "exposition_max": round(float(exp[gs:ge + 1].max()), 1),
            })

    return {
        "portfolio": pf,
        "stats_globales": {
            "total_deces": int(total_d),
            "total_exposition": int(total_e),
            "taux_brut_global_permille": round(taux_brut_global * 1000, 2),
            "age_moyen_pondere": round(age_moyen, 1),
            "age_min": int(ages.min()),
            "age_max": int(ages.max()),
        },
        "lissage": {"methode": lis["methode"], "parametres": lis["parametres"]},
        "validation": val,
        "deciles": dec,
        "echantillon_ages": echantillon,
        "zones_surmortalite": zones_supra,
        "zones_sous_mortalite": zones_infra,
        "zones_faible_exposition": zones_faible,
        "qualite_donnees": qd,
    }


# ─── Narration fallback (mécanique) ───────────────────────────────────────────

def generate_narratif_fallback(payload: dict) -> dict:
    """Narratif mécanique minimal — mode dégradé sans LLM.

    Produit un dict structurellement correct avec des f-strings simples.
    """
    pf = payload["portfolio"]
    db = payload["donnees_brutes"]
    lis = payload["lissage"]
    val = payload["validation"]
    dec = payload["deciles"]
    qd = payload["qualite_donnees"]

    ages = np.array(db["ages"], dtype=float)
    exp = np.array(db["exposure"], dtype=float)
    deaths = np.array(db["deaths_observed"], dtype=float)
    q_lisse = np.array(lis["q_lisse"], dtype=float)
    q_ref = np.array(lis["q_ref"], dtype=float)
    alpha = np.array(payload["abattement"]["alpha"], dtype=float)

    total_d = int(deaths.sum())
    total_e = int(exp.sum())
    smr = val["smr_global"]
    abat = val["abattement_global"]
    methode = lis["methode"].replace("_", " ").title()
    pvalue = val["chi2_pvalue"]
    rej = "ne rejette pas" if pvalue > 0.05 else "rejette"
    isup = "inférieure" if smr < 1 else "supérieure"

    age_moyen = float(np.average(ages, weights=exp)) if exp.sum() > 0 else float(ages.mean())

    alertes = []
    if pvalue > 0.99:
        alertes.append(
            f"p-valeur du χ² très élevée ({pvalue:.3f}) — possible sur-lissage (λ trop grand)."
        )
    for d in dec:
        if d["smr_ic_inf"] > 1:
            alertes.append(
                f"Décile {d['tranche_label']} : SMR={d['smr']:.3f}, IC exclut 1 "
                f"[{d['smr_ic_inf']:.3f};{d['smr_ic_sup']:.3f}] — surmortalité significative."
            )
        elif d["smr_ic_sup"] < 1:
            alertes.append(
                f"Décile {d['tranche_label']} : SMR={d['smr']:.3f}, IC exclut 1 "
                f"[{d['smr_ic_inf']:.3f};{d['smr_ic_sup']:.3f}] — sous-mortalité significative."
            )
    if abat > 1.5:
        alertes.append(
            f"Abattement global {abat:.3f} > 1.5 — mortalité nettement supérieure à la référence."
        )
    elif abat < 0.3:
        alertes.append(
            f"Abattement global {abat:.3f} < 0.3 — à vérifier, possible erreur de référence."
        )

    volume_note = ""
    if total_d < 100:
        volume_note = f" Le volume de {total_d} décès est faible ; les conclusions sont fragiles."
    elif total_d < 500:
        volume_note = f" Le volume de {total_d} décès est modéré ; une analyse de sensibilité est recommandée."

    return {
        "preambule": (
            f"La présente étude porte sur {pf['n_assures']:,} assurés, "
            f"{pf['n_contrats_actifs']:,} contrats actifs, "
            f"période {pf['periode_debut']} au {pf['periode_fin']}. "
            f"{total_d:,} décès ont été observés pour une exposition de {total_e:,} années-personnes. "
            f"Le SMR global s'établit à {smr:.3f} "
            f"[{val['smr_ic_inf']:.3f} ; {val['smr_ic_sup']:.3f}], "
            f"indiquant une mortalité {isup} à la référence {pf['table_reference']}. "
            f"L'abattement global est de {abat:.3f}."
            + volume_note
        ),
        "section_1_contrats": {
            "paragraphes": [
                f"Le portefeuille comprend {pf['n_contrats_actifs']:,} contrats actifs de type "
                f"{pf['type_contrat'].replace('_', ' ')}, segmentés par {pf['segmentation']}. "
                f"La plage d'âges observée s'étend de {pf['age_min']} à {pf['age_max']} ans. "
                f"La table de référence retenue est {pf['table_reference']}.",
                f"L'étude couvre la période du {pf['periode_debut']} au {pf['periode_fin']}, "
                f"soit {len(qd['stats_annuelles'])} années d'observation. "
                f"L'âge moyen pondéré par l'exposition est de {age_moyen:.1f} ans.",
            ]
        },
        "section_2_donnees": {
            "paragraphes_avant_tableaux": [
                f"Les données transmises portent sur {pf['n_assures']:,} assurés. "
                f"Le tableau ci-dessous récapitule les traitements appliqués et les statistiques annuelles.",
            ],
            "paragraphes_apres_tableaux": [
                f"La figure 1 illustre la distribution de l'exposition par âge, "
                f"avec les frontières des déciles d'exposition superposées.",
            ],
        },
        "section_3_methodologie": {
            "intro": (
                f"Les taux bruts sont estimés par le rapport Dx/Ex pour chaque âge x. "
                f"Les âges à exposition inférieure à 10 années-personnes sont exclus de la figure."
            ),
            "commentaire_lissage": (
                f"La méthode de lissage retenue est {methode}. "
                f"Les paramètres sont : {lis['parametres']}. "
                f"Ce lissage minimise un critère pénalisant à la fois "
                f"l'écart aux données brutes et l'irrégularité de la courbe."
            ),
            "commentaire_smr": (
                f"Le SMR global {smr:.3f} [{val['smr_ic_inf']:.3f} ; {val['smr_ic_sup']:.3f}] "
                f"mesure le rapport entre décès observés et attendus sous la table de référence. "
                f"La formule de Liddell est utilisée pour les intervalles de confiance à 95 %."
            ),
            "commentaire_chi2": (
                f"Le test du χ² (Z={val['chi2_stat']:.1f}, {val['chi2_ddl']} ddl, "
                f"p={pvalue:.3f}) {rej} l'adéquation au seuil 5 %."
            ),
            "commentaire_abattement": (
                f"L'abattement αx = qx(exp)/qx(ref) compare âge par âge la table construite "
                f"à la table de référence. Un abattement < 1 signifie une mortalité inférieure. "
                f"L'abattement global est de {abat:.3f}."
            ),
            "commentaire_deciles": (
                f"Les déciles sont construits par quantiles d'exposition cumulée (~10 % chacun), "
                f"non par tranches d'âge fixes. Ceci garantit une puissance statistique "
                f"homogène entre tranches."
            ),
        },
        "section_4_construction": {
            "intro_taux_bruts": (
                f"Le tableau 4 présente les taux bruts q̂x = Dx/Ex pour un échantillon d'âges. "
                f"Les taux en ‰ varient de "
                f"{float(q_lisse.min()) * 1000:.2f} à {float(q_lisse.max()) * 1000:.2f} ‰."
            ),
            "commentaire_taux_lisses": (
                f"Le tableau 5 compare les taux lissés aux taux de référence. "
                f"Les intervalles de confiance à 95 % sont basés sur une approximation de Poisson."
            ),
            "commentaire_figure_taux": (
                f"La figure 2 superpose taux bruts (nuage de points), taux lissés (courbe), "
                f"table de référence (tirets) et IC 95 % (plage). "
                f"La cohérence entre la courbe lissée et la référence reflète l'abattement."
            ),
            "intro_abattement": (
                f"L'abattement αx = qx/qref est calculé pour les âges à exposition suffisante "
                f"(qref > 0,5 ‰). L'abattement global pondéré par l'exposition est {abat:.3f}."
            ),
            "commentaire_figure_abattement": (
                f"La figure 3 montre l'abattement par âge. "
                f"Les zones en rouge indiquent une surmortalité locale (αx ≥ 1), "
                f"en vert une sous-mortalité (αx < 1)."
            ),
        },
        "section_5_commentaires": {
            "paragraphes": [
                f"Le SMR global de {smr:.3f} indique une mortalité {isup} à la référence.",
                (
                    f"Le test d'adéquation χ² (p={pvalue:.3f}) {rej} l'hypothèse de bonne "
                    f"adéquation au seuil 5 %."
                    + (" La valeur très élevée suggère un sur-lissage." if pvalue > 0.99 else "")
                ),
                (
                    f"Le tableau 7 présente les SMR par décile d'exposition. "
                    f"{len([d for d in dec if d['smr_ic_inf'] > 1 or d['smr_ic_sup'] < 1])} "
                    f"décile(s) présentent un SMR significativement différent de 1."
                ),
            ],
            "alertes": alertes,
        },
        "section_6_conclusion": {
            "synthese": (
                f"L'étude porte sur {total_d:,} décès et {total_e:,} AP d'exposition. "
                f"L'abattement global de {abat:.3f} par rapport à {pf['table_reference']} "
                f"et le SMR de {smr:.3f} caractérisent la mortalité du portefeuille."
                + volume_note
            ),
            "recommandations": (
                "Recommandations : (i) réévaluation dans 5 ans ou après 200 décès supplémentaires ; "
                "(ii) surveillance des déciles présentant un SMR significatif ; "
                "(iii) vérification de la cohérence en cas de changement de souscription."
                if total_d >= 100
                else
                f"Volume insuffisant ({total_d} décès) pour des recommandations robustes. "
                "Réévaluation dans 3 ans ou après 100 décès supplémentaires."
            ),
            "validation": f"Table validée sous réserve des recommandations ci-dessus.",
        },
    }


# ─── Point d'entrée piloté par Brief ──────────────────────────────────────────

def generate_mortality_report_from_brief(
    brief_dict: dict,
    payload: dict,
    output_path: str = "rapport_certification.pdf",
    llm_client: Any | None = None,
) -> str:
    """Génère le PDF en tenant compte du Brief produit par le WriterAgent.

    Contrairement à generate_mortality_report(), cette fonction :
    - Vérifie que le payload satisfait les sections obligatoires du brief
    - Passe le brief comme contexte au LLM narrateur (via WriterAgent)

    Args:
        brief_dict:  Dict Brief produit par WriterAgent.conduct_dialog().
        payload:     Dict conforme au data contract.
        output_path: Chemin du PDF de sortie.
        llm_client:  Si fourni, activé pour la narration (sinon fallback).

    Returns:
        Chemin absolu du PDF généré.

    Raises:
        ValueError: Si payload invalide ou sections obligatoires manquantes.
    """
    brief = Brief.from_dict(brief_dict)

    # Vérifier que le payload satisfait les sections obligatoires du brief
    missing = check_payload_against_brief(brief, payload)
    if missing:
        raise ValueError(
            f"Payload incomplet — sections obligatoires du brief non satisfaites : "
            f"{missing}"
        )

    # Validation standard du payload
    result = validate(payload)
    if not result.valid:
        raise ValueError(result.refusal_message())

    # Narration : via WriterAgent (brief-aware) ou fallback
    if llm_client is not None:
        from report_agent.writer_agent import WriterAgent
        writer = WriterAgent()
        narratif = writer.generate_narrative(brief, payload)
    else:
        narratif = generate_narratif_fallback(payload)

    return render_pdf(narratif, payload, output_path)
