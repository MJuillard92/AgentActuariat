#!/usr/bin/env python3
"""
ingest_doctrine.py — Construit l'index FAISS du corpus doctrinal actuariel
====================================================================

Lit les 142 chunks enrichis (chunks_enriched.json, versionné), les embeds via
bge-m3 (multilingue, local, gratuit) et écrit l'index FAISS + meta JSON dans
`knowledge_base/rag_doctrine/index/`.

Lancer ce script :
  - Au premier checkout du repo (l'index n'est PAS versionné)
  - Après tout update du fichier chunks_enriched.json

Dépendances : faiss-cpu, sentence-transformers, torch, rank-bm25 (pour BM25
côté retrieval, pas nécessaire ici à l'ingestion). Voir requirements.txt.

Premier run : télécharge BAAI/bge-m3 (~2.3 Go) depuis HuggingFace Hub,
mis en cache dans ~/.cache/huggingface/. Suivants : instantané.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from tools.conversation._retriever._pack_embed import (
    HFEmbedder, build_index, save_index,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> int:
    here = Path(__file__).parent
    chunks_path = here / "chunks_enriched.json"
    index_dir   = here / "index"
    index_path  = index_dir / "faiss.bin"
    meta_path   = index_dir / "meta.json"

    if not chunks_path.exists():
        log.error(f"chunks_enriched.json absent : {chunks_path}")
        return 1

    with chunks_path.open(encoding="utf-8") as f:
        chunks = json.load(f)
    log.info(f"Chunks chargés : {len(chunks)}")

    texts = [c["text"] for c in chunks]

    # MiniLM multilingue : 120 Mo, dim 384, charge en ~1s vs ~5-10s pour
    # bge-m3. Recall@5 légèrement inférieur (~-5pts sur les requêtes
    # techniques) mais perceptiblement meilleur côté UX.
    # Pour passer à bge-m3 : remplacer ci-dessous par HFEmbedder("BAAI/bge-m3").
    log.info("Chargement embedder MiniLM multilingue (1er run = download ~120 Mo)…")
    emb = HFEmbedder("paraphrase-multilingual-MiniLM-L12-v2")
    log.info(f"Embedder prêt : {emb.name}, dim={emb.dim}")

    log.info("Embedding des 142 chunks…")
    vectors = emb.embed(texts)
    log.info(f"Vectors : shape={vectors.shape}, dtype={vectors.dtype}")

    log.info("Construction de l'index FAISS (IndexFlatIP, cosine)…")
    index = build_index(vectors)

    index_dir.mkdir(parents=True, exist_ok=True)
    save_index(index, index_path)
    log.info(f"Index FAISS écrit : {index_path}")

    meta = {
        "embedder": emb.name,
        "dim":      emb.dim,
        "n":        len(chunks),
        "chunks":   chunks,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info(f"Meta JSON écrit : {meta_path}")
    log.info("OK — l'index est prêt pour conversation.search_doctrine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
