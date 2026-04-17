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
    baseline_regulatory_table:   Optional[str]  = None   # "TH0002"
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

        # Paramètres d'étude
        sp_dict = {k: v for k, v in self.study_plan.model_dump().items() if v is not None}
        if sp_dict:
            ds["study_plan"] = sp_dict

        # Référence dataset
        if self.dataset_meta:
            ds["_dataset_ref"] = self.session_id

        # Résultats tools
        ds.update(self.tool_results)

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

        # study_plan
        sp_data = data_store.get("study_plan")
        if sp_data and isinstance(sp_data, dict):
            # Merge : ne pas écraser les champs déjà définis avec None
            current = self.study_plan.model_dump()
            merged  = {k: v for k, v in {**current, **sp_data}.items()}
            self.study_plan = StudyPlan(**merged)

        # Résultats tools (uniquement les clés métier connues, JSON-safe)
        _TOOL_RESULT_KEYS = {
            "exposure_table", "qx_table", "smoothed_table", "diagnostics",
            "validation", "benchmarking", "certification_report", "summary",
            "cox_regression", "logit_regression", "series",
        }
        for key in _TOOL_RESULT_KEYS:
            if data_store.get(key) is not None:
                self.tool_results[key] = data_store[key]

        self.touch()
