"""
retriever.py — Retriever hybride dense (FAISS) + sparse (BM25) avec RRF.

Fonctionnalites :
- Recherche dense via embeddings + FAISS (cosine via IndexFlatIP normalise L2)
- Recherche sparse BM25 (rank_bm25)
- Fusion Reciprocal Rank Fusion (RRF, k=60 par defaut)
- Filtrage par metadonnees (doc_id, tags, regulatory, tables_referenced)
- Reranking optionnel via cross-encoder BAAI/bge-reranker-v2-m3

Usage:
    from retriever import HybridRetriever
    r = HybridRetriever.from_paths("index/faiss.bin", "index/meta.json",
                                    embedder="bge-m3")
    results = r.retrieve("Test du chi-2 sur tables d'experience",
                          k=5, filters={"tags": ["validation"]})
"""

from __future__ import annotations
import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""
    section_id: str = ""
    section_title: str = ""
    rank_dense: Optional[int] = None
    rank_sparse: Optional[int] = None
    rerank_score: Optional[float] = None


def _tokenize_fr(text: str) -> list[str]:
    """Tokenisation simple FR + chiffres + identifiants A132-18."""
    text = text.lower()
    tokens = re.findall(r"[a-z\u00e0\u00e2\u00e4\u00e9\u00e8\u00ea\u00eb\u00ef\u00ee\u00f4\u00f6\u00f9\u00fb\u00fc\u00e7\u00b5\u03bc0-9][a-z\u00e0\u00e2\u00e4\u00e9\u00e8\u00ea\u00eb\u00ef\u00ee\u00f4\u00f6\u00f9\u00fb\u00fc\u00e70-9\-]*", text)
    return [t for t in tokens if len(t) >= 2]


class HybridRetriever:
    def __init__(
        self,
        index,
        chunks: list[dict],
        embedder,
        bm25,
        tokenized_corpus: list[list[str]],
    ):
        self.index = index
        self.chunks = chunks
        self.embedder = embedder
        self.bm25 = bm25
        self.tokenized_corpus = tokenized_corpus
        self._reranker = None

    @classmethod
    def from_paths(
        cls,
        index_path: str | Path,
        meta_path: str | Path,
        embedder: str = "bge-m3",
    ) -> "HybridRetriever":
        import faiss
        from rank_bm25 import BM25Okapi
        from ._pack_embed import get_embedder

        index = faiss.read_index(str(index_path))
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        chunks = meta["chunks"]

        emb = get_embedder(embedder)
        if hasattr(emb, "prefix") and emb.prefix:
            emb.prefix = "query: "  # pour e5 cote requete
        tokenized = [_tokenize_fr(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        return cls(index, chunks, emb, bm25, tokenized)

    def _dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        qv = self.embedder.embed([query])
        qv = qv / max(np.linalg.norm(qv), 1e-12)
        scores, idxs = self.index.search(qv.astype(np.float32), k)
        return [(int(i), float(s)) for s, i in zip(scores[0], idxs[0]) if i != -1]

    def _sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        tokens = _tokenize_fr(query)
        scores = self.bm25.get_scores(tokens)
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    @staticmethod
    def _rrf(rankings: list[list[tuple[int, float]]], k_rrf: int = 60) -> dict[int, float]:
        """Reciprocal Rank Fusion."""
        agg: dict[int, float] = {}
        for ranking in rankings:
            for rank, (idx, _score) in enumerate(ranking):
                agg[idx] = agg.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
        return agg

    def _filter_chunk(self, chunk: dict, filters: dict) -> bool:
        md = chunk.get("metadata", {})
        if "doc_id" in filters:
            ids = filters["doc_id"]
            if isinstance(ids, str):
                ids = [ids]
            if chunk.get("doc_id") not in ids:
                return False
        if "tags" in filters:
            required = set(filters["tags"])
            if not required.issubset(set(md.get("tags", []))):
                return False
        if filters.get("regulatory") is True and not md.get("regulatory", False):
            return False
        if "tables_referenced" in filters:
            req = set(filters["tables_referenced"])
            present = set(md.get("tables_referenced", []))
            if not req.issubset(present):
                return False
        return True

    def _load_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
        return self._reranker

    def retrieve(
        self,
        query: str,
        k: int = 5,
        k_initial: int = 30,
        filters: Optional[dict] = None,
        rerank: bool = False,
    ) -> list[RetrievalResult]:
        dense = self._dense_search(query, k_initial)
        sparse = self._sparse_search(query, k_initial)
        agg = self._rrf([dense, sparse])

        dense_rank = {idx: r for r, (idx, _) in enumerate(dense)}
        sparse_rank = {idx: r for r, (idx, _) in enumerate(sparse)}

        ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        results: list[RetrievalResult] = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            if filters and not self._filter_chunk(chunk, filters):
                continue
            results.append(
                RetrievalResult(
                    chunk_id=chunk["chunk_id"],
                    score=score,
                    text=chunk["text"],
                    metadata=chunk.get("metadata", {}),
                    doc_id=chunk.get("doc_id", ""),
                    section_id=chunk.get("section_id", ""),
                    section_title=chunk.get("section_title", ""),
                    rank_dense=dense_rank.get(idx),
                    rank_sparse=sparse_rank.get(idx),
                )
            )
            if len(results) >= max(k, k_initial // 2):
                break

        if rerank and results:
            ranker = self._load_reranker()
            pairs = [(query, r.text) for r in results]
            scores = ranker.predict(pairs).tolist()
            for r, s in zip(results, scores):
                r.rerank_score = float(s)
            results.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)

        return results[:k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--embedder", default="bge-m3")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--tags", default=None)
    parser.add_argument("--regulatory", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level="INFO")

    r = HybridRetriever.from_paths(args.index, args.meta, embedder=args.embedder)
    filters: dict = {}
    if args.doc_id:
        filters["doc_id"] = args.doc_id
    if args.tags:
        filters["tags"] = [t.strip() for t in args.tags.split(",")]
    if args.regulatory:
        filters["regulatory"] = True

    res = r.retrieve(args.query, k=args.k, filters=filters or None, rerank=args.rerank)
    for i, hit in enumerate(res, 1):
        print(f"\n=== Hit {i}  score={hit.score:.4f}"
              + (f"  rerank={hit.rerank_score:.4f}" if hit.rerank_score else "")
              + f"  [{hit.doc_id} / {hit.section_id}] {hit.section_title}")
        print(hit.text[:500] + ("..." if len(hit.text) > 500 else ""))


if __name__ == "__main__":
    main()
