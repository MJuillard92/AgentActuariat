"""
session/session_state.py
Schéma canonique de l'état de session — source de vérité métier.

Couches :
  BusinessMemory  → SessionState (ce fichier)
  WorkingMemory   → AgentState LangGraph (state.py)
  ConversationMem → messages LangGraph (tronqués/compactés)
  AuditLog        → sessions/{id}_audit.json (append-only)

Le SessionState est la seule source persistée sur disque qui fait foi.
LangGraph MemorySaver sert de cache RAM entre les tours d'une même session.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Sous-modèles métier ────────────────────────────────────────────────────────

class StudyPlan(BaseModel):
    """Paramètres d'étude actuarielle validés par l'utilisateur."""
    observation_start_date:      Optional[str]  = None   # "YYYY-MM-DD"
    observation_end_date:        Optional[str]  = None
    observation_period_years:    List[int]       = Field(default_factory=list)
    study_objective:             Optional[str]  = None
    cohort_min_age:              Optional[int]  = None
    cohort_max_age:              Optional[int]  = None
    smoothing_algorithm:         Optional[str]  = None   # "whittaker_henderson"
    smoothing_parameters:        Optional[str]  = None
    baseline_regulatory_table:   Optional[str]  = None   # "TH0002"
    product_list:                Optional[str]  = None
    exclusion_criteria:          Optional[str]  = None
    boundary_age_treatment:      Optional[str]  = None
    confidence_interval_level:   float          = 0.95
    chi_squared_p_significance:  float          = 0.05

    def is_complete(self) -> bool:
        required = [
            self.observation_start_date,
            self.observation_end_date,
            self.baseline_regulatory_table,
        ]
        return all(v is not None for v in required)


class DatasetMeta(BaseModel):
    """Référence stable vers l'artefact DataFrame — écrite une seule fois."""
    path:       str             # sessions/artifacts/{session_id}_dataset.parquet
    sha256:     str             # 12 premiers chars du hash
    n_rows:     int
    n_cols:     int
    columns:    List[str]
    created_at: str             # ISO datetime

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_cols)


class ContextSummary(BaseModel):
    """
    Résumé structuré de la conversation compactée.
    Remplace l'historique verbatim au-delà de COMPACT_THRESHOLD messages.
    """
    decisions_prises:    List[str] = Field(default_factory=list)
    ambiguites_levees:   List[str] = Field(default_factory=list)
    hypotheses_actives:  List[str] = Field(default_factory=list)
    objets_construits:   List[str] = Field(default_factory=list)  # clés data_store
    donnees_manquantes:  List[str] = Field(default_factory=list)
    prochaine_etape:     str       = ""
    compacted_at:        str       = ""   # ISO datetime de la dernière compaction
    messages_since:      int       = 0    # messages verbatim depuis la compaction

    def to_system_block(self) -> str:
        """Retourne un bloc texte à injecter dans le system prompt."""
        lines = ["## Contexte antérieur (résumé structuré)"]
        if self.decisions_prises:
            lines.append("**Décisions prises :** " + " | ".join(self.decisions_prises))
        if self.ambiguites_levees:
            lines.append("**Ambiguïtés levées :** " + " | ".join(self.ambiguites_levees))
        if self.hypotheses_actives:
            lines.append("**Hypothèses actives :** " + " | ".join(self.hypotheses_actives))
        if self.objets_construits:
            lines.append("**Calculs réalisés :** " + ", ".join(self.objets_construits))
        if self.donnees_manquantes:
            lines.append("**Données manquantes :** " + ", ".join(self.donnees_manquantes))
        if self.prochaine_etape:
            lines.append(f"**Prochaine étape :** {self.prochaine_etape}")
        return "\n".join(lines)


# ── Modèle principal ──────────────────────────────────────────────────────────

class SessionState(BaseModel):
    """
    État canonique d'une session — persiste dans sessions/{id}_state.json.

    Règle d'or : ne jamais stocker de données brutes volumineuses ici.
    - Le DataFrame → DatasetMeta.path (Parquet, écrit une fois)
    - Les résultats intermédiaires de tools → tool_results (JSON-safe)
    """
    session_id:  str
    version:     int  = 1
    created_at:  str  = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at:  str  = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    csv_filename: Optional[str] = None

    # ── Objets métier ─────────────────────────────────────────────────────────
    study_plan:               StudyPlan               = Field(default_factory=StudyPlan)
    column_mapping:           Dict[str, str]           = Field(default_factory=dict)
    column_mapping_confirmed: bool                     = False
    column_mapping_unmatched: List[str]                = Field(default_factory=list)
    disambiguation_done:      bool                     = False

    # ── Référence dataset (écrite une seule fois) ─────────────────────────────
    dataset_meta:  Optional[DatasetMeta]  = None

    # ── Résultats tools validés ───────────────────────────────────────────────
    # Seules les clés JSON-safe (pas de DataFrames raw) :
    # exposure_table, qx_table, smoothed_table, validation, benchmarking…
    tool_results:  Dict[str, Any]  = Field(default_factory=dict)

    # ── Mémoire conversationnelle compactée ───────────────────────────────────
    context_summary:  Optional[ContextSummary]  = None

    # ── Flags cinématiques préservés entre tours ──────────────────────────────
    # Ces clés alimentent les branches de routage du Master (need_user_input,
    # préservation _write/report_mode, anti-boucle, accumulateur user_messages).
    # Sans cette persistance, elles sont perdues entre deux invocations
    # successives du graphe LangGraph et la cinématique perd la mémoire.
    cinematic_state: Dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = datetime.datetime.now().isoformat()

    def to_data_store(self) -> dict:
        """
        Hydrate le data_store LangGraph depuis ce SessionState.
        Appelé au début de chaque stream_agent() pour initialiser le state LangGraph.
        """
        ds: dict = {}

        # Flags disambiguation
        if self.disambiguation_done:
            ds["_disambiguation_done"]     = True
            ds["column_mapping"]           = self.column_mapping
            ds["column_mapping_confirmed"] = self.column_mapping_confirmed
            ds["column_mapping_unmatched"] = self.column_mapping_unmatched

        # Paramètres d'étude — avec dérivation automatique des champs calendaires
        sp = self.study_plan
        if sp.observation_start_date and sp.observation_end_date and not sp.observation_period_years:
            try:
                start_y = int(sp.observation_start_date[:4])
                end_y   = int(sp.observation_end_date[:4])
                sp.observation_period_years = list(range(start_y, end_y + 1))
            except (ValueError, TypeError):
                pass

        sp_dict = {k: v for k, v in sp.model_dump().items() if v is not None and v != []}
        if sp_dict:
            ds["study_plan"] = sp_dict
            # Exposer aussi au niveau racine du data_store pour le WriterAgent
            if sp.observation_period_years:
                ds["observation_period_years"] = sp.observation_period_years
                ds["num_observation_years"]     = len(sp.observation_period_years)

        # Référence dataset
        if self.dataset_meta:
            ds["_dataset_ref"] = self.session_id

        # Résultats tools
        ds.update(self.tool_results)

        # Flags cinématiques (préservés entre tours)
        ds.update(self.cinematic_state)

        return ds

    def update_from_data_store(self, data_store: dict) -> None:
        """
        Extrait les objets métier depuis le data_store LangGraph
        et met à jour ce SessionState.
        """
        # Flags
        if data_store.get("_disambiguation_done"):
            self.disambiguation_done      = True
            self.column_mapping           = data_store.get("column_mapping") or self.column_mapping
            self.column_mapping_confirmed = data_store.get("column_mapping_confirmed", self.column_mapping_confirmed)
            self.column_mapping_unmatched = data_store.get("column_mapping_unmatched", self.column_mapping_unmatched)

        # study_plan — merge aussi les scalaires exposure qui matchent des champs StudyPlan
        sp_data = data_store.get("study_plan") or {}
        for key in ("cohort_min_age", "cohort_max_age"):
            if data_store.get(key) is not None and key not in sp_data:
                sp_data[key] = data_store[key]
        if sp_data:
            current = self.study_plan.model_dump()
            merged  = {k: v for k, v in {**current, **sp_data}.items()}
            self.study_plan = StudyPlan(**merged)

        # Résultats tools (uniquement les clés métier connues, JSON-safe)
        _TOOL_RESULT_KEYS = {
            "exposure_table", "qx_table", "smoothed_table", "diagnostics",
            "validation", "benchmarking", "certification_report", "summary",
            "cox_regression", "logit_regression", "series",
            # Scalaires issus du tool exposure — perdus entre les tours sans cette liste
            "total_deaths", "total_exposure_years", "cohort_min_age", "cohort_max_age",
            "age_min", "age_max", "total_exposure", "n_insured",
        }
        for key in _TOOL_RESULT_KEYS:
            if data_store.get(key) is not None:
                self.tool_results[key] = data_store[key]

        # Flags cinématiques préservés entre tours (stricte liste blanche).
        # IMPORTANT : on synchronise dans LES DEUX SENS — si une clé a été
        # consommée (data_store.pop) pendant le tour, elle doit aussi
        # disparaître de la persistance. Sinon le tour suivant la verrait
        # ressusciter via to_data_store().
        _CINEMATIC_KEYS = {
            "_pending_need",
            "_user_messages",
            "_master_builder_cycles",
            "_questions_asked_this_cycle",
            "_write",
            "_kind",
            "report_mode",
            "_write_question_asked",
        }
        for key in _CINEMATIC_KEYS:
            if key in data_store:
                self.cinematic_state[key] = data_store[key]
            else:
                # Pop côté persistance pour rester cohérent avec le data_store
                self.cinematic_state.pop(key, None)

        self.touch()
