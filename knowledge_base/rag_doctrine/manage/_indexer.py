"""
_indexer.py — Reconstruction atomique de l'index FAISS du corpus doctrinal.

Pipeline :
  chunks_enriched.json
    → HFEmbedder("paraphrase-multilingual-MiniLM-L12-v2").embed(texts)
    → faiss.IndexFlatIP (cosine via L2 normalize)
    → écriture atomique : faiss.bin.tmp + meta.json.tmp puis os.replace

Le nom de l'embedder est figé : changer d'embedder casserait le retriever
en cache RAM (mismatch de dimension). Vérification par assert.

IO utilities :
  - load_chunks / save_chunks_with_backup (avec .bak.YYYYMMDD-HHMMSS)
  - sha256_file (fingerprint pour idempotence)
  - existing_fingerprints / next_doc_id (helpers métier)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Permet l'import direct du module quand le script est exécuté en standalone
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

log = logging.getLogger(__name__)

# Constantes
EMBEDDER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDER_DIM = 384

# Chemins canoniques du corpus doctrine
_DOCTRINE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = _DOCTRINE_DIR / "chunks_enriched.json"
INDEX_DIR = _DOCTRINE_DIR / "index"
INDEX_PATH = INDEX_DIR / "faiss.bin"
META_PATH = INDEX_DIR / "meta.json"


# ─────────────────────────────────────────────────────────────────────────────
# IO chunks
# ─────────────────────────────────────────────────────────────────────────────
def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    """Charge chunks_enriched.json. Retourne [] si fichier absent."""
    if not path.exists():
        log.warning("chunks file absent : %s — retour liste vide", path)
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_chunks_with_backup(chunks: list[dict], path: Path = CHUNKS_PATH) -> Path | None:
    """Sauvegarde JSON avec backup horodaté.

    Retourne le chemin du .bak (None si pas de fichier précédent).
    Écriture atomique via .tmp + os.replace.
    """
    bak_path: Path | None = None
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        bak_path = path.with_suffix(path.suffix + f".bak.{ts}")
        shutil.copy2(path, bak_path)
        log.info("Backup : %s", bak_path)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    log.info("chunks écrits : %s (%d chunks)", path, len(chunks))
    return bak_path


def sha256_file(path: Path, chunk_size: int = 65536) -> str:
    """SHA256 hex digest d'un fichier (lecture chunked)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def existing_fingerprints(chunks: list[dict]) -> dict[str, str]:
    """Retourne {source_fingerprint: doc_id} pour les chunks ayant un fingerprint."""
    out: dict[str, str] = {}
    for c in chunks:
        fp = (c.get("metadata") or {}).get("source_fingerprint")
        doc_id = c.get("doc_id")
        if fp and doc_id and fp not in out:
            out[fp] = doc_id
    return out


_DOC_ID_RE = re.compile(r"^D(\d{2,3})$")


def next_doc_id(chunks: list[dict]) -> str:
    """Auto-incrémente le doc_id depuis l'existant. Format D{NN}."""
    max_n = 0
    for c in chunks:
        m = _DOC_ID_RE.match(c.get("doc_id", ""))
        if m:
            max_n = max(max_n, int(m.group(1)))
    nxt = max_n + 1
    if nxt > 999:
        raise NotImplementedError("Capacité doc_id dépassée (>D999)")
    return f"D{nxt:02d}" if nxt < 100 else f"D{nxt:03d}"


def utcnow_iso() -> str:
    """Timestamp UTC ISO 8601 sans microsecondes (ex: 2026-05-19T14:32:11Z)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ─────────────────────────────────────────────────────────────────────────────
# Rebuild index FAISS
# ─────────────────────────────────────────────────────────────────────────────
def rebuild_index(
    chunks_path: Path = CHUNKS_PATH,
    index_dir: Path = INDEX_DIR,
    embedder_name: str = EMBEDDER_NAME,
) -> None:
    """Reconstruit l'index FAISS depuis chunks_enriched.json.

    Écriture atomique : on écrit faiss.bin.tmp + meta.json.tmp et on remplace
    via os.replace() à la fin, pour qu'un consommateur concurrent ne voie
    jamais un état partiel.

    Si chunks vide → écrit un index vide cohérent (utile après suppression totale).
    """
    from tools.conversation._retriever._pack_embed import (
        HFEmbedder, build_index, save_index,
    )

    chunks = load_chunks(chunks_path)
    log.info("Rebuild index : %d chunks", len(chunks))

    log.info("Chargement embedder %s …", embedder_name)
    emb = HFEmbedder(embedder_name)
    assert emb.dim == EMBEDDER_DIM, (
        f"Embedder dim mismatch : {emb.dim} attendu {EMBEDDER_DIM}. "
        f"Changer d'embedder casserait le retriever."
    )

    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "faiss.bin"
    meta_path = index_dir / "meta.json"
    tmp_index = index_path.with_suffix(".bin.tmp")
    tmp_meta = meta_path.with_suffix(".json.tmp")

    if chunks:
        texts = [c["text"] for c in chunks]
        log.info("Embedding de %d chunks …", len(texts))
        vectors = emb.embed(texts)
        log.info("Vectors shape=%s dtype=%s", vectors.shape, vectors.dtype)
        index = build_index(vectors)
    else:
        # Index vide cohérent : on crée un IndexFlatIP sans vecteurs
        import faiss
        import numpy as np
        log.warning("Aucun chunk : création d'un index vide (dim=%d)", emb.dim)
        index = faiss.IndexFlatIP(emb.dim)
        # Pas d'add → index.ntotal == 0
        vectors = np.zeros((0, emb.dim), dtype="float32")  # noqa: F841

    save_index(index, tmp_index)
    meta = {
        "embedder": emb.name,
        "dim":      emb.dim,
        "n":        len(chunks),
        "chunks":   chunks,
    }
    with tmp_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    os.replace(tmp_index, index_path)
    os.replace(tmp_meta, meta_path)
    log.info("Index reconstruit : %s + %s", index_path, meta_path)
