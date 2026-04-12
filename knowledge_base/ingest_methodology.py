#!/usr/bin/env python3
"""
ingest_methodology.py — Ingestion de la documentation méthodologique dans ChromaDB.

Lit les fichiers JSON de la base de connaissance méthodologique et les ingère
dans la collection "documentation_actuarielle" de ChromaDB.

Usage :
    # Ingérer les fichiers JSON par défaut (Knowledge Base/04_smoothing.json, etc.)
    python base_de_connaissance/ingest_methodology.py

    # Ingérer un fichier JSON spécifique
    python base_de_connaissance/ingest_methodology.py --json chemin/vers/fichier.json

    # Ingérer depuis un répertoire de PDF (extraction texte + chunking)
    python base_de_connaissance/ingest_methodology.py --pdf-dir base_de_connaissance/methodologie/raw/

    # Afficher l'état de la collection sans rien ingérer
    python base_de_connaissance/ingest_methodology.py --status

    # Vider la collection et réingérer depuis zéro
    python base_de_connaissance/ingest_methodology.py --reset

Format JSON attendu (liste de chunks) :
    [
        {
            "id"      : "chunk_001",         (ou "chunk_id")
            "contenu" : "Texte du chunk...", (ou "content")
            "source"  : "Seance6.pdf",
            "section" : "2.3 Whittaker-Henderson",
            "titre"   : "Paramètre lambda",
            "type"    : "methodologie"
        },
        ...
    ]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT   = Path(__file__).resolve().parent.parent
_DEFAULT_CHROMA = _PROJECT_ROOT / "base_de_connaissance" / "methodologie" / "chromadb"
_DEFAULT_RAW    = _PROJECT_ROOT / "base_de_connaissance" / "methodologie" / "raw"
_COLLECTION     = "documentation_actuarielle"

# Fichiers JSON candidats (cherchés dans l'ordre)
_JSON_CANDIDATES = [
    _PROJECT_ROOT / "Knowledge Base" / "04_smoothing.json",
    _PROJECT_ROOT / "lissage_chunks_complet.json",
    _PROJECT_ROOT / "Knowledge Base" / "lissage_chunks_complet.json",
    _DEFAULT_RAW,  # répertoire de fallback (sera scanné si c'est un dossier)
]


# ─── helpers ──────────────────────────────────────────────────────────────────

def _get_collection(chroma_path: Path, collection_name: str = _COLLECTION):
    """Retourne la collection ChromaDB (crée si absente)."""
    try:
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    except ImportError:
        sys.exit("Erreur : chromadb non installé. pip install chromadb")

    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    ef     = DefaultEmbeddingFunction()

    try:
        return client.get_collection(name=collection_name, embedding_function=ef)
    except Exception:
        return client.get_or_create_collection(name=collection_name, embedding_function=ef)


def _chunk_id(item: dict, fallback_index: int) -> str:
    cid = item.get("id") or item.get("chunk_id")
    if cid:
        return str(cid)
    contenu = item.get("contenu") or item.get("content") or ""
    return "chunk_" + hashlib.md5(contenu.encode()).hexdigest()[:12]


def _parse_json_file(path: Path) -> list[dict]:
    """Lit un fichier JSON de chunks méthodologiques."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        print(f"  Ignoré (pas une liste) : {path.name}")
        return []
    return raw


def _chunks_from_json(items: list[dict], source_file: str) -> tuple[list, list, list]:
    """Transforme une liste de dicts JSON en (ids, docs, metas)."""
    ids, docs, metas = [], [], []
    for i, item in enumerate(items):
        contenu = item.get("contenu") or item.get("content") or ""
        if not contenu.strip():
            continue
        cid  = _chunk_id(item, i)
        meta = {
            "source":  item.get("source", source_file),
            "section": item.get("section", "inconnu"),
            "titre":   item.get("titre", item.get("title", "")),
            "type":    item.get("type", "methodologie"),
        }
        ids.append(cid)
        docs.append(contenu)
        metas.append(meta)
    return ids, docs, metas


# ─── ingestion depuis un PDF ──────────────────────────────────────────────────

def _extract_and_chunk_pdf(pdf_path: Path, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """Extrait le texte d'un PDF et le découpe en chunks."""
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    pages.append(t.strip())
        full_text = "\n\n".join(pages)
    except ImportError:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            full_text = "\n\n".join(
                p.extract_text() or "" for p in reader.pages
            )
        except ImportError:
            print(f"  Ignoré (pdfplumber/pypdf non installé) : {pdf_path.name}")
            return []

    # Normaliser
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", full_text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
                current = current[-overlap:].strip() if overlap else ""
            current = para

    if current:
        chunks.append(current)

    result = []
    for i, content in enumerate(chunks):
        h = hashlib.md5(content.encode()).hexdigest()[:8]
        result.append({
            "id":      f"{pdf_path.stem}_chunk_{i:04d}_{h}",
            "contenu": content,
            "source":  pdf_path.name,
            "section": "inconnu",
            "titre":   "",
            "type":    "methodologie",
        })
    return result


# ─── commandes ────────────────────────────────────────────────────────────────

def cmd_status(chroma_path: Path) -> None:
    coll  = _get_collection(chroma_path)
    count = coll.count()
    print(f"Collection '{_COLLECTION}' : {count} chunks")
    if count > 0:
        sample = coll.get(limit=3, include=["metadatas"])
        for sid, meta in zip(sample["ids"], sample["metadatas"]):
            src = (meta or {}).get("source", "?")
            print(f"  {sid}  [{src}]")


def cmd_reset(chroma_path: Path) -> None:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_path))
        try:
            client.delete_collection(_COLLECTION)
            print(f"Collection '{_COLLECTION}' supprimée.")
        except Exception:
            print(f"Collection '{_COLLECTION}' inexistante, rien à supprimer.")
    except ImportError:
        sys.exit("Erreur : chromadb non installé.")


def cmd_ingest_json(json_path: Path, chroma_path: Path) -> int:
    items = _parse_json_file(json_path)
    if not items:
        print(f"  Aucun chunk valide dans {json_path.name}.")
        return 0

    ids, docs, metas = _chunks_from_json(items, json_path.name)
    if not ids:
        return 0

    coll = _get_collection(chroma_path)
    existing = set(coll.get(include=[])["ids"])

    new_ids   = [i for i in ids if i not in existing]
    new_docs  = [d for i, d in zip(ids, docs) if i not in existing]
    new_metas = [m for i, m in zip(ids, metas) if i not in existing]

    if not new_ids:
        print(f"  Tous les chunks déjà présents ({len(ids)} ignorés).")
        return 0

    coll.add(ids=new_ids, documents=new_docs, metadatas=new_metas)
    print(f"  {len(new_ids)} chunks ingérés ({len(ids) - len(new_ids)} doublons ignorés).")
    return len(new_ids)


def cmd_ingest_pdf_dir(pdf_dir: Path, chroma_path: Path) -> int:
    pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
    if not pdf_files:
        print(f"  Aucun PDF trouvé dans {pdf_dir}")
        return 0

    coll     = _get_collection(chroma_path)
    existing = set(coll.get(include=[])["ids"])
    total    = 0

    for pdf_path in sorted(pdf_files):
        print(f"  PDF : {pdf_path.name}")
        items = _extract_and_chunk_pdf(pdf_path)
        if not items:
            continue
        ids, docs, metas = _chunks_from_json(items, pdf_path.name)
        new_ids   = [i for i in ids if i not in existing]
        new_docs  = [d for i, d in zip(ids, docs) if i not in existing]
        new_metas = [m for i, m in zip(ids, metas) if i not in existing]
        if new_ids:
            coll.add(ids=new_ids, documents=new_docs, metadatas=new_metas)
            existing.update(new_ids)
            print(f"    {len(new_ids)} chunks ingérés.")
            total += len(new_ids)
        else:
            print(f"    Tous les chunks déjà présents.")

    return total


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingestion de la documentation méthodologique dans ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json",       default=None, help="Chemin vers un fichier JSON à ingérer")
    parser.add_argument("--pdf-dir",    default=None, help="Répertoire de PDF à ingérer")
    parser.add_argument("--status",     action="store_true", help="Afficher l'état de la collection")
    parser.add_argument("--reset",      action="store_true", help="Vider la collection avant ingestion")
    parser.add_argument("--chroma-path", default=str(_DEFAULT_CHROMA))

    args = parser.parse_args()
    chroma_path = Path(args.chroma_path)

    if args.status:
        cmd_status(chroma_path)
        return

    if args.reset:
        cmd_reset(chroma_path)

    if args.json:
        json_path = Path(args.json)
        if not json_path.exists():
            sys.exit(f"Erreur : fichier introuvable : {json_path}")
        print(f"Ingestion depuis : {json_path}")
        n = cmd_ingest_json(json_path, chroma_path)
        print(f"Total : {n} chunks ingérés.")
        return

    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.is_dir():
            sys.exit(f"Erreur : répertoire introuvable : {pdf_dir}")
        print(f"Ingestion des PDF depuis : {pdf_dir}")
        n = cmd_ingest_pdf_dir(pdf_dir, chroma_path)
        print(f"Total : {n} chunks ingérés.")
        return

    # Mode par défaut : parcourir les candidats JSON connus
    print("Ingestion depuis les fichiers JSON par défaut…")
    total = 0
    found = False
    for candidate in _JSON_CANDIDATES:
        if isinstance(candidate, Path) and candidate.is_file():
            print(f"\n→ {candidate.relative_to(_PROJECT_ROOT)}")
            n = cmd_ingest_json(candidate, chroma_path)
            total += n
            found = True
        elif isinstance(candidate, Path) and candidate.is_dir():
            json_files = list(candidate.glob("*.json"))
            for jf in sorted(json_files):
                print(f"\n→ {jf.relative_to(_PROJECT_ROOT)}")
                n = cmd_ingest_json(jf, chroma_path)
                total += n
                found = True

    if not found:
        print(
            "Aucun fichier JSON trouvé dans les emplacements par défaut.\n"
            "Utiliser --json <chemin> pour spécifier un fichier."
        )
        return

    print(f"\nTotal : {total} chunks ingérés dans '{_COLLECTION}'.")
    cmd_status(chroma_path)


if __name__ == "__main__":
    main()
