"""
embed_and_index.py — Embeddings + index FAISS pour AgentActuariat.

Embedders supportes :
- 'openai'     : OpenAI text-embedding-3-large (3072 dim)
- 'bge-m3'     : BAAI/bge-m3 (1024 dim, multilingue)
- 'me5-large'  : intfloat/multilingual-e5-large (1024 dim)

Index FAISS avec normalisation L2 (cosine similarity).

Usage:
    python embed_and_index.py --input chunks_enriched.json \\
        --embedder bge-m3 --index ./index/faiss.bin --meta ./index/meta.json
"""

from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    name: str = "base"
    dim: int = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class OpenAIEmbedder(Embedder):
    name = "openai-text-embedding-3-large"
    dim = 3072

    def __init__(self, model: str = "text-embedding-3-large", api_key: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        BATCH = 100
        vecs = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vecs.extend([d.embedding for d in resp.data])
        return np.asarray(vecs, dtype=np.float32)


class HFEmbedder(Embedder):
    """Embedder HuggingFace generique (sentence-transformers)."""
    def __init__(self, model_name: str, prefix: str = ""):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()
        self.prefix = prefix  # ex "passage: " pour e5

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.prefix:
            texts = [self.prefix + t for t in texts]
        arr = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            batch_size=32,
            show_progress_bar=False,
        )
        return arr.astype(np.float32)


def get_embedder(kind: str) -> Embedder:
    kind = kind.lower()
    if kind == "openai":
        return OpenAIEmbedder()
    if kind == "bge-m3":
        return HFEmbedder("BAAI/bge-m3")
    if kind == "me5-large":
        return HFEmbedder("intfloat/multilingual-e5-large", prefix="passage: ")
    raise ValueError(f"Embedder inconnu: {kind}")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return x / n


def build_index(vectors: np.ndarray):
    import faiss
    vectors = l2_normalize(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])  # cosine via dot product
    index.add(vectors)
    return index


def save_index(index, path: Path):
    import faiss
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--embedder", default="bge-m3", choices=["openai", "bge-m3", "me5-large"])
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    with args.input.open(encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)

    texts = [c["text"] for c in chunks]
    embedder = get_embedder(args.embedder)
    logger.info("Embedder: %s (dim=%d), %d textes a encoder.",
                embedder.name, embedder.dim, len(texts))

    vectors = embedder.embed(texts)
    logger.info("Vectors shape: %s", vectors.shape)

    index = build_index(vectors)
    save_index(index, args.index)
    logger.info("Index FAISS sauvegarde : %s", args.index)

    meta = {
        "embedder": embedder.name,
        "dim": embedder.dim,
        "n": len(chunks),
        "chunks": chunks,
    }
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    with args.meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("Meta JSON sauvegarde : %s", args.meta)


if __name__ == "__main__":
    main()
