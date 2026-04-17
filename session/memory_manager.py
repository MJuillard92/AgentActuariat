"""
session/memory_manager.py
Orchestrateur des 4 couches mémoire.

  BusinessMemory  → SessionState  (session/data/{id}_state.json)
  WorkingMemory   → AgentState LangGraph (MemorySaver RAM)
  ConversationMem → messages tronqués + ContextSummary injecté en system
  AuditLog        → session/data/{id}_audit.json (append-only, géré par canvas_app)

Interface publique :
  mm = MemoryManager(session_id)
  mm.load()                           ← charger depuis disque
  ds = mm.to_data_store()             ← hydrater le state LangGraph
  mm.after_turn(data_store, messages) ← persister après chaque tour
  ctx = mm.get_context_block()        ← bloc system prompt (summary + dataset meta)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from session.session_state import SessionState, DatasetMeta, StudyPlan, ContextSummary
from session.dataset_store import DatasetStore
from session.summarizer import Summarizer

log = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent / "data"


class MemoryManager:
    """
    Gère le cycle de vie complet de la mémoire d'une session.

    Usage typique dans graph.py / stream_agent() :
        mm = MemoryManager(session_id)
        mm.load()
        initial_data_store = mm.to_data_store()
        # ... stream LangGraph ...
        mm.after_turn(data_store, messages)
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._state_path = _DATA_DIR / f"{session_id}_state.json"
        self._summarizer = Summarizer()
        self.state: SessionState = SessionState(session_id=session_id)

    # ── Persistance ───────────────────────────────────────────────────────────

    def load(self) -> "MemoryManager":
        """
        Charge le SessionState depuis disque.
        Si le fichier n'existe pas, initialise un SessionState vide.
        """
        if self._state_path.exists():
            try:
                raw = json.loads(self._state_path.read_text(encoding="utf-8"))
                self.state = SessionState.model_validate(raw)
                log.debug("[MemoryManager] session %s chargée — %d tool_results",
                          self.session_id, len(self.state.tool_results))
            except Exception as exc:
                log.warning("[MemoryManager] échec chargement %s : %s — état vide",
                            self._state_path, exc)
                self.state = SessionState(session_id=self.session_id)
        return self

    def save(self) -> None:
        """Persiste le SessionState sur disque (JSON indenté)."""
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            self._state_path.write_text(
                self.state.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            log.error("[MemoryManager] échec sauvegarde %s : %s", self._state_path, exc)

    # ── Interface avec LangGraph ──────────────────────────────────────────────

    def to_data_store(self) -> dict:
        """
        Produit le data_store initial pour le state LangGraph.
        Appelé en début de stream_agent() pour hydrater l'AgentState.
        """
        return self.state.to_data_store()

    def after_turn(self, data_store: dict, messages: list | None = None) -> None:
        """
        Appelé après chaque tour LangGraph (fin de stream_agent()).

        1. Extrait les objets métier depuis data_store → SessionState
        2. Déclenche la compaction si nécessaire
        3. Persiste sur disque
        """
        self.state.update_from_data_store(data_store)

        if messages and self._summarizer.should_compact(messages):
            log.info("[MemoryManager] compaction déclenchée — %d messages", len(messages))
            summary = self._summarizer.compact(messages, data_store)
            self.state.context_summary = summary

        self.save()

    # ── Helpers pour les nodes ────────────────────────────────────────────────

    def get_context_block(self) -> str:
        """
        Retourne un bloc texte à injecter en fin de system prompt.
        Contient : résumé structuré + métadonnées dataset.
        """
        blocks: list[str] = []

        if self.state.context_summary:
            blocks.append(self.state.context_summary.to_system_block())

        if self.state.dataset_meta:
            meta = self.state.dataset_meta
            blocks.append(
                f"\n## Dataset chargé\n"
                f"- **{meta.n_rows:,} lignes** × {meta.n_cols} colonnes\n"
                f"- Colonnes : {', '.join(meta.columns)}\n"
                f"- Hash SHA-256 : `{meta.sha256}`"
            )

        return "\n\n".join(blocks)

    def load_dataframe(self):
        """
        Charge le DataFrame depuis l'artefact Parquet.
        Retourne None si aucun dataset n'est enregistré pour cette session.
        """
        if self.state.dataset_meta is None:
            return DatasetStore.load_by_session(self.session_id)
        try:
            return DatasetStore.load(self.state.dataset_meta)
        except FileNotFoundError as exc:
            log.warning("[MemoryManager] %s", exc)
            return None

    # ── Enregistrement dataset ────────────────────────────────────────────────

    def register_dataset(self, df, csv_filename: str | None = None) -> DatasetMeta:
        """
        Enregistre le DataFrame initial (une seule fois).
        Idempotent : si déjà enregistré, retourne le DatasetMeta existant.

        Args:
            df           : pandas DataFrame
            csv_filename : nom du fichier source (pour affichage)

        Returns:
            DatasetMeta
        """
        if self.state.dataset_meta is not None:
            # Déjà enregistré — ne pas réécrire
            log.debug("[MemoryManager] dataset déjà enregistré pour session %s", self.session_id)
            return self.state.dataset_meta

        meta = DatasetStore.store(self.session_id, df)
        self.state.dataset_meta = meta
        if csv_filename:
            self.state.csv_filename = csv_filename
        self.save()
        log.info("[MemoryManager] dataset enregistré — %d×%d, hash=%s",
                 meta.n_rows, meta.n_cols, meta.sha256)
        return meta

    # ── Compaction explicite (optionnel) ──────────────────────────────────────

    def trim_messages(self, messages: list) -> list:
        """
        Si le contexte_summary existe, retourne les N messages verbatim récents.
        Sinon retourne les messages inchangés.
        """
        if self.state.context_summary and self._summarizer.should_compact(messages):
            return self._summarizer.trim_messages(messages)
        return messages
