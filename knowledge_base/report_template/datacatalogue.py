"""
knowledge_base/report_template/datacatalogue.py — Gate prérequis Builder.

PRINCIPE ARCHITECTURAL (plan 2026-05-25) :
Ce helper est DOMAIN-AGNOSTIC. Il ne sait rien des taux de mortalité,
du lissage, ni d'aucune méthode actuarielle. Il agrège mécaniquement trois
sources de vérité :

  Source A — YAML `mortality_template.yaml` (ou toute autre template) :
    entrées `data_contract.master_from_*` portant `confirm_with_user: true`.
    Ajouter un nouveau prérequis = ajouter une ligne YAML. Aucun code Python.

  Source B — Catalogue des tools (TOOL CONTRACT docstrings) :
    via `method_choices_for_mode()`, retourne les choix de méthodes
    encore non résolus pour le `report_mode` courant.

  Source C — Flag UI `mapping_validated` :
    posé par le bouton « Valider le mapping » dans canvas_app.

Le Builder appelle `compute_datacatalogue_state(data_store)` au tout début
de son nœud : si `complete=False`, il refuse net et le Master ouvre le
panneau UI « Compléter le data catalogue ».

Quand on ajoutera d'autres agents de calcul (provisionnement, Solvabilité,
etc.), ils livreront leur propre template YAML — ce helper continuera de
fonctionner sans modification de code Master/Builder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from knowledge_base.report_template.template_loader import build_manifest


@dataclass
class DataCatalogueState:
    """Résultat du diagnostic des prérequis utilisateur.

    Attributs :
      complete : True si AUCUN champ requis n'est manquant.
      missing  : liste des clés manquantes (ex: `gender_segmentation`,
                 `methods.builder.smoothing`, `mapping_validated`).
      state    : snapshot {clé: valeur} pour debug et rendu UI.
    """
    complete: bool
    missing:  list[str] = field(default_factory=list)
    state:    dict[str, Any] = field(default_factory=dict)


def _is_empty(value: Any) -> bool:
    """Considère un champ manquant si valeur None, chaîne vide, liste/dict
    vide. False est une réponse VALIDE (ex. write=no), pas un manquant.
    """
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict, tuple, set)) and len(value) == 0:
        return True
    return False


def compute_datacatalogue_state(
    data_store: dict | None,
    template_path=None,
) -> DataCatalogueState:
    """Diagnostic complet : tous les prérequis utilisateur sont-ils
    renseignés pour que le Builder puisse tourner ?

    Args:
        data_store : état LangGraph courant (lecture seule, pas de mutation).
        template_path : path optionnel vers une template YAML alternative
            (pour tests, ou futurs domaines non-mortalité).

    Returns:
        DataCatalogueState. Le caller (Builder ou UI) inspecte `complete`
        et `missing`.
    """
    data_store = data_store or {}
    sp = data_store.get("study_plan") or {}
    missing: list[str] = []
    state: dict[str, Any] = {}

    # ── Source C — Flag mapping CSV ──────────────────────────────────────
    if not data_store.get("mapping_validated"):
        missing.append("mapping_validated")
        state["mapping_validated"] = False
    else:
        state["mapping_validated"] = True

    # ── Source A — YAML : champs confirm_with_user: true ─────────────────
    # Lecture dynamique : si demain on ajoute une entrée YAML avec
    # confirm_with_user, elle est automatiquement détectée ici.
    try:
        manifest = build_manifest(template_path) if template_path \
            else build_manifest()
    except Exception:
        # Template introuvable ou parsing échoué : on remonte tout comme
        # manquant pour éviter de laisser passer un Builder à l'aveugle.
        return DataCatalogueState(
            complete=False,
            missing=["template_yaml_unreadable"],
            state={"template_error": True},
        )

    for spec in (*manifest.master_from_data, *manifest.master_from_modeling):
        if not spec.confirm_with_user:
            continue
        # Lookup hiérarchique : study_plan d'abord, puis data_store
        # top-level (compat avec l'historique).
        value = sp.get(spec.key)
        if _is_empty(value):
            value = data_store.get(spec.key)
        state[spec.key] = value
        if _is_empty(value):
            missing.append(spec.key)

    # ── Source B — Catalogue tools : choix de méthodes manquants ─────────
    # Sémantique :
    #   - methods_auto=True   → user a explicitement délégué → OK
    #   - methods=<dict>      → user a explicitement commencé à préciser →
    #                            on vérifie qu'il a fini (tous les tools)
    #   - les DEUX absents    → user n'a rien dit → on assume `auto` par
    #                            défaut (pas de friction inutile). L'user
    #                            peut toujours forcer un choix explicite
    #                            via le bouton sidebar « Compléter le data
    #                            catalogue » (option « Préciser »).
    report_mode = data_store.get("report_mode") or sp.get("report_mode")
    gender = sp.get("gender_segmentation") or data_store.get("gender_segmentation")
    methods_picked = sp.get("methods") or {}
    if (report_mode
            and not sp.get("methods_auto")
            and methods_picked):
        # User a commencé à choisir manuellement — on vérifie qu'il a fini.
        try:
            from agents.master.method_choices import method_choices_for_mode
            remaining = method_choices_for_mode(report_mode, gender, sp)
        except Exception:
            remaining = []
        for choice in remaining:
            key = f"methods.{choice.tool}"
            missing.append(key)
            state[key] = None

    return DataCatalogueState(
        complete=(len(missing) == 0),
        missing=missing,
        state=state,
    )
