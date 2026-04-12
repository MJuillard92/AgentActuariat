#!/usr/bin/env python3
"""
chunk_report.py — Ingestion d'un rapport PDF dans le corpus exemplaires ChromaDB.

Usage :
    python base_de_connaissance/chunk_report.py \
        --pdf chemin/vers/rapport.pdf \
        --rapport-id "CERT-2024-HOM-001" \
        --type-rapport "certification" \
        --produit "temporaire_deces" \
        --methode-lissage "whittaker" \
        --periode "2018-2023" \
        --qualite "high"

Options :
    --pdf             (requis) Chemin vers le PDF à ingérer
    --rapport-id      Identifiant unique du rapport (ex : CERT-2024-001)
    --type-rapport    Type : "certification" | "construction" | "actualisation" | "audit"
    --produit         Produit couvert (ex : "temporaire_deces", "rente_viagere")
    --methode-lissage Méthode de lissage utilisée (ex : "whittaker", "gompertz", "splines")
    --periode         Période d'observation (ex : "2018-2023")
    --qualite         Qualité du rapport : "high" | "medium" | "low"
    --chunk-size      Taille cible d'un chunk en caractères (défaut : 600)
    --overlap         Chevauchement entre chunks en caractères (défaut : 80)
    --dry-run         Afficher les chunks sans ingérer
    --collection      Nom de la collection ChromaDB (défaut : exemplaires_actuariels)
    --chroma-path     Chemin ChromaDB (défaut : base_de_connaissance/exemplaires/chromadb)
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CHROMA = _PROJECT_ROOT / "base_de_connaissance" / "exemplaires" / "chromadb"
_DEFAULT_RAW    = _PROJECT_ROOT / "base_de_connaissance" / "exemplaires" / "raw"
_DEFAULT_COLLECTION = "exemplaires_actuariels"


# ─── extraction PDF ───────────────────────────────────────────────────────────

def extract_text(pdf_path: Path) -> str:
    """Extrait le texte d'un PDF avec pdfplumber (page par page, séparé par ---PAGE---).
    Fallback sur pypdf si pdfplumber n'est pas installé."""
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"[PAGE {i}]\n{text.strip()}")
        return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages  = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[PAGE {i}]\n{text.strip()}")
        return "\n\n".join(pages)
    except ImportError:
        sys.exit(
            "Erreur : ni pdfplumber ni pypdf n'est installé.\n"
            "Installer avec : pip install pdfplumber  ou  pip install pypdf"
        )


# ─── nettoyage ────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Supprime les artefacts PDF courants et normalise les espaces."""
    # Supprimer les en-têtes/pieds de page courts répétitifs (< 60 chars, ligne seule)
    text = re.sub(r"\n[ \t]*[^\n]{1,60}[ \t]*\n(?=\n)", "\n\n", text)
    # Normaliser les espaces multiples
    text = re.sub(r" {2,}", " ", text)
    # Normaliser les sauts de ligne multiples
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─── découpage en chunks ──────────────────────────────────────────────────────

def make_chunks(
    text: str,
    chunk_size: int = 600,
    overlap: int = 80,
    rapport_id: str = "",
) -> list[dict]:
    """
    Découpe le texte en chunks chevauchants.
    Respecte les séparations de paragraphes quand possible.
    Retourne une liste de dicts {chunk_id, contenu, position}.
    """
    # Séparer par paragraphes
    paragraphs = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[dict] = []
    current = ""
    para_idx = 0

    while para_idx < len(paragraphs):
        para = paragraphs[para_idx]

        if len(current) + len(para) + 1 <= chunk_size:
            current = (current + "\n\n" + para).strip()
            para_idx += 1
        else:
            if current:
                chunks.append(current)
                # Chevauchement : garder les derniers `overlap` chars du chunk courant
                current = current[-overlap:].strip() if overlap else ""
            else:
                # Paragraphe trop long → découper brutalement par chunk_size
                for start in range(0, len(para), chunk_size - overlap):
                    chunk = para[start: start + chunk_size]
                    if chunk.strip():
                        chunks.append(chunk.strip())
                para_idx += 1
                current = ""

    if current:
        chunks.append(current)

    # Construire la liste de dicts avec IDs déterministes
    result = []
    for i, content in enumerate(chunks):
        # ID stable basé sur le contenu
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
        chunk_id = f"{rapport_id}_chunk_{i:04d}_{content_hash}" if rapport_id else f"chunk_{i:04d}_{content_hash}"
        result.append({
            "chunk_id":  chunk_id,
            "contenu":   content,
            "position":  i,
        })

    return result


# ─── ingestion ChromaDB ───────────────────────────────────────────────────────

def ingest_chunks(
    chunks: list[dict],
    metadata_base: dict,
    collection_name: str,
    chroma_path: Path,
) -> int:
    """Ingère les chunks dans ChromaDB. Retourne le nombre de chunks ingérés."""
    try:
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    except ImportError:
        sys.exit(
            "Erreur : chromadb non installé.\n"
            "Installer avec : pip install chromadb"
        )

    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    ef     = DefaultEmbeddingFunction()

    try:
        collection = client.get_collection(name=collection_name, embedding_function=ef)
    except Exception:
        collection = client.get_or_create_collection(name=collection_name, embedding_function=ef)

    # Vérifier les IDs existants pour éviter les doublons
    existing_ids = set(collection.get(include=[])["ids"])

    ids, docs, metas = [], [], []
    skipped = 0
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if cid in existing_ids:
            skipped += 1
            continue
        meta = {**metadata_base, "position": chunk["position"]}
        ids.append(cid)
        docs.append(chunk["contenu"])
        metas.append(meta)

    if ids:
        collection.add(ids=ids, documents=docs, metadatas=metas)

    if skipped:
        print(f"  {skipped} chunks déjà présents ignorés.")

    return len(ids)


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingestion d'un rapport PDF dans le corpus exemplaires ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pdf",             required=True,  help="Chemin vers le PDF")
    parser.add_argument("--rapport-id",      default="",     help="Identifiant unique du rapport")
    parser.add_argument("--type-rapport",    default="",     help="Type de rapport (certification, construction, ...)")
    parser.add_argument("--produit",         default="",     help="Produit d'assurance couvert")
    parser.add_argument("--methode-lissage", default="",     help="Méthode de lissage utilisée")
    parser.add_argument("--periode",         default="",     help="Période d'observation")
    parser.add_argument("--qualite",         default="medium", choices=["high", "medium", "low"])
    parser.add_argument("--chunk-size",      type=int, default=600)
    parser.add_argument("--overlap",         type=int, default=80)
    parser.add_argument("--dry-run",         action="store_true", help="Afficher les chunks sans ingérer")
    parser.add_argument("--collection",      default=_DEFAULT_COLLECTION)
    parser.add_argument("--chroma-path",     default=str(_DEFAULT_CHROMA))

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"Erreur : fichier introuvable : {pdf_path}")

    rapport_id = args.rapport_id or pdf_path.stem

    print(f"Extraction du texte depuis : {pdf_path.name}")
    raw_text = extract_text(pdf_path)
    clean    = clean_text(raw_text)
    print(f"  {len(clean):,} caractères extraits.")

    print(f"Découpage en chunks (taille={args.chunk_size}, overlap={args.overlap})…")
    chunks = make_chunks(clean, args.chunk_size, args.overlap, rapport_id)
    print(f"  {len(chunks)} chunks générés.")

    if args.dry_run:
        print("\n─── DRY RUN — premiers 3 chunks ───")
        for chunk in chunks[:3]:
            print(f"\n[{chunk['chunk_id']}]\n{chunk['contenu'][:300]}…\n")
        return

    # Copier le PDF dans raw/
    raw_dir = _DEFAULT_RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / pdf_path.name
    if not dest.exists():
        import shutil
        shutil.copy2(str(pdf_path), str(dest))
        print(f"  PDF copié vers : {dest.relative_to(_PROJECT_ROOT)}")

    metadata_base = {
        "rapport_id":          rapport_id,
        "type_rapport":        args.type_rapport,
        "produit":             args.produit,
        "methode_lissage":     args.methode_lissage,
        "periode_observation": args.periode,
        "qualite_exemplaire":  args.qualite,
        "source":              pdf_path.name,
    }

    chroma_path = Path(args.chroma_path)
    print(f"Ingestion dans ChromaDB : {chroma_path.relative_to(_PROJECT_ROOT)}")
    n_ingeres = ingest_chunks(chunks, metadata_base, args.collection, chroma_path)
    print(f"  {n_ingeres} chunks ingérés avec succès.")
    print(f"\nRapport '{rapport_id}' ajouté au corpus exemplaires.")


if __name__ == "__main__":
    main()
