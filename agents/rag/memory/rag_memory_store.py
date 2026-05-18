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
from agents.rag.memory.summarizer import summarize_old_turns

log = logging.getLogger(__name__)

# ── Sizing parameters (référencés dans tout le pipeline) ────────────────────
BUFFER_SIZE             = 4
SUMMARY_TRIGGER         = 10
VECTORSTORE_TOP_K       = 3
VECTORSTORE_MIN_SCORE   = 0.7
MAX_MEMORY_CHARS        = 4000


def _sanitize_for_memory(text: str) -> str:
    """Nettoie un texte avant insertion en mémoire conversationnelle.

    1. Truncate à MAX_MEMORY_CHARS
    2. Strip control chars (préserve \\n \\t)
    3. Neutralise les marqueurs structurels (évite pollution multi-tour
       si user injecte '[Conversation récente]' dans sa question)
    """
    if not text:
        return ""
    text = text[:MAX_MEMORY_CHARS]
    text = "".join(c for c in text if c in ("\n", "\t") or ord(c) >= 32)
    for marker in (
        "[Conversation récente]", "[Résumé contexte antérieur]",
        "[Échanges passés pertinents]", "[Nouvelle question]",
        "[Question utilisateur]", "[Extraits doctrinaux]",
    ):
        text = text.replace(marker, "(neutralisé)")
    return text


class RAGMemoryStore:
    """Per-session conversational memory. RAM-only with lazy rebuild."""

    _cache: dict[str, "RAGMemoryStore"] = {}  # module-level cache

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._buffer:        list[RAGTurn] = []
        self._summary:       RAGSummary | None = None
        # Vectorstore (niveau 3) — init paresseuse au premier _index_turn_in_vectorstore.
        # Déclarés ici pour que les lecteurs voient TOUS les attributs d'instance.
        self._faiss_index = None  # IndexFlatIP | None — type lazy import pour éviter faiss à l'import
        self._indexed_turns: list[RAGTurn] = []
        self._total_turns: int = 0   # compteur global pour summary trigger

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
        import faiss

        embedder = self._get_embedder()
        # Init paresseuse de l'index FAISS au premier ajout
        if self._faiss_index is None:
            self._faiss_index = faiss.IndexFlatIP(embedder.dim)

        # Embed le texte concaténé Q + A
        text = f"{turn.user_q}\n{turn.rag_answer}"
        vec = embedder.embed([text])
        # Normalisation L2 pour utiliser le produit scalaire comme cosine
        faiss.normalize_L2(vec)
        # Atomicité : append PUIS add, rollback si add échoue
        self._indexed_turns.append(turn)
        try:
            self._faiss_index.add(vec)
        except Exception:
            # Préserve l'alignement _indexed_turns[i] ↔ FAISS index i
            self._indexed_turns.pop()
            raise

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
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
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

    # ── Lazy rebuild from history + module-level cache ──────────────────

    @classmethod
    def for_session(
        cls,
        session_id: str,
        history: list | None = None,
    ) -> "RAGMemoryStore":
        """Retourne le store de la session (cache hit) ou en crée un nouveau
        en reconstruisant depuis history (cache miss = cold start)."""
        store = cls._cache.get(session_id)
        if store is None:
            store = cls(session_id)
            store._rebuild_from_history(history or [])
            cls._cache[session_id] = store
        return store

    def _rebuild_from_history(self, history: list) -> None:
        """Reconstruit buffer + vectorstore depuis l'historique LangChain.

        Extrait les paires (HumanMessage, AIMessage) consécutives, ignore
        les HumanMessage marqués source='master_synthetic' (relances Master,
        pas de vraies questions user).

        Note : le SUMMARY (niveau 2) n'est PAS reconstruit ici. Au prochain
        passage du seuil SUMMARY_TRIGGER dans append_turn, il sera regénéré
        from scratch sur les tours anciens disponibles. Comportement voulu :
        évite un appel LLM coûteux au cold start, et le summary sera produit
        à la demande quand vraiment utile.
        """
        pairs = self._extract_qa_pairs(history)
        # Buffer : les BUFFER_SIZE dernières paires
        for user_q, rag_answer in pairs[-BUFFER_SIZE:]:
            self._append_buffer_only(user_q, rag_answer, sources=[])
        # Vectorstore : toutes les paires (économie : skip si <2 paires)
        if len(pairs) >= 2:
            for user_q, rag_answer in pairs:
                self._index_turn_in_vectorstore(
                    RAGTurn(user_q=user_q, rag_answer=rag_answer, sources=[])
                )

    @staticmethod
    def _extract_qa_pairs(history: list) -> list[tuple[str, str]]:
        """Extrait les paires user/AI consécutives de l'historique LangChain.

        Ignore les HumanMessage avec additional_kwargs.source='master_synthetic'.
        """
        from langchain_core.messages import HumanMessage, AIMessage
        pairs: list[tuple[str, str]] = []
        pending_user: str | None = None
        for m in history:
            if isinstance(m, HumanMessage):
                kwargs = getattr(m, "additional_kwargs", None) or {}
                if kwargs.get("source") == "master_synthetic":
                    pending_user = None
                    continue
                pending_user = (m.content or "").strip() or None
            elif isinstance(m, AIMessage) and pending_user:
                ai_content = (m.content or "").strip()
                if ai_content:
                    pairs.append((pending_user, ai_content))
                pending_user = None
        return pairs

    # ── API publique : écriture (utilisée par run_pipeline RAG.7) ───────

    def append_turn(self, user_q: str, rag_answer: str,
                    sources: list[dict] | None = None) -> None:
        """Ajoute un tour à la mémoire conversationnelle.

        1. Sanitize (truncate + strip + neutralize markers)
        2. Append au buffer (fifo)
        3. Index dans le vectorstore (embedding MiniLM)
        4. Trigger summary update si total_turns > SUMMARY_TRIGGER

        Note : append_turn assume que le tour fourni n'est PAS déjà dans le
        vectorstore (sinon double-indexing). Le caller (run_pipeline RAG.7)
        garantit que append_turn n'est appelé qu'une fois par nouveau tour,
        APRÈS que for_session(history) a populé la mémoire depuis l'history
        passé (qui ne contient pas encore le tour courant).
        """
        user_q_clean     = _sanitize_for_memory(user_q)
        rag_answer_clean = _sanitize_for_memory(rag_answer)
        sources          = sources or []

        # 1. Buffer (réutilise _append_buffer_only pour cohérence FIFO)
        self._append_buffer_only(user_q_clean, rag_answer_clean, sources=sources)

        # 2. Vectorstore (réutilise _index_turn_in_vectorstore)
        self._index_turn_in_vectorstore(
            RAGTurn(user_q=user_q_clean, rag_answer=rag_answer_clean,
                    sources=sources)
        )

        # 3. Compteur
        self._total_turns += 1

        # 4. Trigger summary (strict > pour éviter compaction sur 0 ancien)
        if self._total_turns > SUMMARY_TRIGGER:
            old_turns = self._get_old_turns_for_summary()
            self._summary = summarize_old_turns(old_turns, existing=self._summary)

    def _get_old_turns_for_summary(self) -> list[RAGTurn]:
        """Retourne les tours indexés au vectorstore qui ne sont pas dans le buffer.

        Le buffer contient les BUFFER_SIZE derniers ; on prend les indexed_turns
        plus anciens que ces BUFFER_SIZE derniers pour la compaction.
        """
        if not self._indexed_turns:
            return []
        if len(self._indexed_turns) <= BUFFER_SIZE:
            return []
        return self._indexed_turns[:-BUFFER_SIZE]
