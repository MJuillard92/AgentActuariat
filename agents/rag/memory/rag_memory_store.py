"""
RAGMemoryStore — Mémoire conversationnelle 3-niveaux par session, RAM-only.

**ÉTAT v0.5 (Task 5)** : seul le niveau 1 (buffer ring-fifo) est opérationnel.
Niveaux 2 (summary) et 3 (vectorstore FAISS) livrés en Tasks 6/7/10.

Niveau 1 : buffer ring-fifo des derniers BUFFER_SIZE tours (verbatim)
Niveau 2 : RAGSummary mis à jour synchronement après SUMMARY_TRIGGER tours
Niveau 3 : index FAISS de tous les Q/A embedded (MiniLM-384)

Aucune persistence disque. Lazy rebuild depuis le `history` LangGraph au
premier accès (cold start après restart Flask = ~1s pour 20 Q/A).

Cache module-level par session_id. Pas de TTL en v1.
"""
from __future__ import annotations

import logging

from agents.rag.memory.schemas import RAGTurn, RAGSummary

log = logging.getLogger(__name__)

# ── Sizing parameters (référencés dans tout le pipeline) ────────────────────
BUFFER_SIZE             = 4
SUMMARY_TRIGGER         = 10
VECTORSTORE_TOP_K       = 3
VECTORSTORE_MIN_SCORE   = 0.7
MAX_MEMORY_CHARS        = 4000


class RAGMemoryStore:
    """Per-session conversational memory. RAM-only with lazy rebuild."""

    _cache: dict[str, "RAGMemoryStore"] = {}  # module-level cache

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._buffer:   list[RAGTurn] = []
        self._summary:  RAGSummary | None = None
        # vectorstore + ses méthodes arrivent à la Task 6

    # ── Public API : lectures ────────────────────────────────────────────

    def get_buffer(self, n: int = BUFFER_SIZE) -> list[RAGTurn]:
        """Retourne les n derniers tours du buffer (limité à BUFFER_SIZE)."""
        return self._buffer[-n:] if n else []

    def get_summary(self) -> RAGSummary | None:
        """Retourne le summary courant (None si jamais généré)."""
        return self._summary

    # ── Méthode interne (testée à part) — sera enrobée par append_turn ──

    def _append_buffer_only(self, user_q: str, rag_answer: str,
                             sources: list[dict] | None = None) -> None:
        """Ajoute au buffer avec éviction FIFO. Pas de vectorstore ni summary."""
        turn = RAGTurn(user_q=user_q, rag_answer=rag_answer, sources=sources or [])
        self._buffer.append(turn)
        if len(self._buffer) > BUFFER_SIZE:
            self._buffer = self._buffer[-BUFFER_SIZE:]

    # ── Vectorstore (niveau 3) ───────────────────────────────────────────

    # Cache module-level de l'embedder (partagé entre toutes les sessions).
    # MiniLM-384 ~120 Mo, chargé une seule fois.
    _embedder = None

    @classmethod
    def _get_embedder(cls):
        if cls._embedder is None:
            from tools.conversation._retriever._pack_embed import get_embedder
            cls._embedder = get_embedder("minilm")
        return cls._embedder

    def _index_turn_in_vectorstore(self, turn: RAGTurn) -> None:
        """Embed le tuple (user_q + rag_answer) et l'ajoute à l'index FAISS."""
        import numpy as np
        import faiss

        # Stockage paresseux : init au premier ajout
        if not hasattr(self, "_faiss_index") or self._faiss_index is None:
            embedder = self._get_embedder()
            dim = embedder.dim
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._indexed_turns: list[RAGTurn] = []

        embedder = self._get_embedder()
        # Embed le texte concaténé Q + A
        text = f"{turn.user_q}\n{turn.rag_answer}"
        vec = embedder.embed([text])
        # Normalisation L2 pour utiliser le produit scalaire comme cosine
        faiss.normalize_L2(vec)
        self._faiss_index.add(vec)
        self._indexed_turns.append(turn)

    def retrieve_similar(
        self,
        query: str,
        k: int = VECTORSTORE_TOP_K,
        min_score: float = VECTORSTORE_MIN_SCORE,
    ) -> list[RAGTurn]:
        """Retourne les top-k tours sémantiquement similaires à la query.

        Filtre par min_score (cosine similarity ∈ [-1, 1]). Retourne liste
        vide si l'index est vide ou si aucun hit ne dépasse min_score.
        """
        if not getattr(self, "_faiss_index", None) or self._faiss_index.ntotal == 0:
            return []
        import faiss
        embedder = self._get_embedder()
        qvec = embedder.embed([query])
        faiss.normalize_L2(qvec)
        k_safe = min(k, self._faiss_index.ntotal)
        scores, indices = self._faiss_index.search(qvec, k_safe)
        results: list[RAGTurn] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS pad avec -1 si pas assez de résultats
                continue
            if float(score) < min_score:
                continue
            results.append(self._indexed_turns[idx])
        return results
