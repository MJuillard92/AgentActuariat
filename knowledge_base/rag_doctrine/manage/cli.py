"""
cli.py — CLI de gestion du corpus doctrinal RAG.

Sous-commandes :
  add      : injecte les PDF/DOCX d'un dossier dans chunks_enriched.json + rebuild index
  list     : liste les documents et leurs chunks
  delete   : supprime un doc_id (et tous ses chunks)
  rebuild  : reconstruit l'index FAISS depuis chunks_enriched.json (sans modif JSON)

Usage :
  python -m knowledge_base.rag_doctrine.manage.cli add --dir /path/to/folder
  python -m knowledge_base.rag_doctrine.manage.cli list
  python -m knowledge_base.rag_doctrine.manage.cli list --doc-id D03 --format json
  python -m knowledge_base.rag_doctrine.manage.cli delete --doc-id D13 --confirm
  python -m knowledge_base.rag_doctrine.manage.cli rebuild

Les fonctions métier (ingest_files, delete_doc, list_docs, build_chunks_from_file)
sont importables depuis ui.py pour servir l'onglet Dash sans dupliquer la logique.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import uuid
from pathlib import Path

from . import _chunking, _enrich, _extract, _indexer

log = logging.getLogger(__name__)

_DOC_ID_PATTERN = re.compile(r"^D\d{2,3}$")
_SUPPORTED_EXT = (".pdf", ".docx")


# ─────────────────────────────────────────────────────────────────────────────
# Logique métier (réutilisable par CLI et UI)
# ─────────────────────────────────────────────────────────────────────────────

def build_chunks_from_file(
    source_path: Path,
    doc_id: str,
    doc_title: str | None = None,
    *,
    target_chars: int = 2000,
    max_chars: int = 2500,
    min_chars: int = 400,
) -> list[dict]:
    """Pipeline complet pour UN fichier : extract → chunk → enrich → assemble.

    Retourne la liste de chunks au schéma chunks_enriched.json.
    Lève ValueError sur extension non supportée.
    """
    fingerprint = _indexer.sha256_file(source_path)
    ingested_at = _indexer.utcnow_iso()
    blocks = _extract.extract(source_path)
    raw_chunks = _chunking.chunk_document(
        blocks, target_chars=target_chars, max_chars=max_chars, min_chars=min_chars,
    )

    final_doc_title = doc_title or f"{doc_id} — {source_path.stem}"

    out: list[dict] = []
    for i, rc in enumerate(raw_chunks, start=1):
        section_id = f"{doc_id}.{i:02d}"
        raw_title = rc.get("section_title_raw")
        section_title = (
            f"{section_id} — {raw_title}" if raw_title else f"{section_id} — Section {i}"
        )
        text = rc["text"]
        metadata = _enrich.enrich_metadata(
            text,
            doc_id=doc_id,
            section_id=section_id,
            source_fingerprint=fingerprint,
            source_filename=source_path.name,
            ingested_at=ingested_at,
        )
        out.append({
            "chunk_id":      str(uuid.uuid4()),
            "doc_id":        doc_id,
            "doc_title":     final_doc_title,
            "section_id":    section_id,
            "section_title": section_title,
            "section_path":  [final_doc_title, section_title],
            "text":          text,
            "word_count":    len(text.split()),
            "block_indices": rc.get("block_indices", []),
            "metadata":      metadata,
        })
    return out


def ingest_files(
    paths: list[Path],
    *,
    forced_doc_id: str | None = None,
    doc_title: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Ingère une liste de fichiers PDF/DOCX dans le corpus doctrine.

    - forced_doc_id : si fourni, doit être unique (validation upstream) ; sinon auto-incrément
    - force : autorise le remplacement si un fingerprint identique est déjà présent
              (supprime d'abord les chunks correspondants)
    - dry_run : aucune écriture (ni JSON ni index), retourne le résumé

    Retourne :
      {
        "added":   [{"file": str, "doc_id": str, "n_chunks": int, "tags": [...]}],
        "skipped": [{"file": str, "reason": str, "existing_doc_id": str|None}],
        "total_chunks_after": int,
      }
    """
    if forced_doc_id and len(paths) > 1:
        raise ValueError("--doc-id ne peut être utilisé qu'avec un seul fichier")
    if forced_doc_id and not _DOC_ID_PATTERN.match(forced_doc_id):
        raise ValueError(f"doc_id invalide : {forced_doc_id} (regex ^D\\d{{2,3}}$)")

    existing = _indexer.load_chunks()
    fingerprints = _indexer.existing_fingerprints(existing)

    added: list[dict] = []
    skipped: list[dict] = []
    new_chunks: list[dict] = []
    cumulative = list(existing)  # mutable copy pour next_doc_id incrémental

    for path in paths:
        if path.suffix.lower() not in _SUPPORTED_EXT:
            skipped.append({"file": path.name, "reason": f"extension non supportée ({path.suffix})", "existing_doc_id": None})
            continue

        fingerprint = _indexer.sha256_file(path)
        if fingerprint in fingerprints:
            if not force:
                skipped.append({
                    "file": path.name,
                    "reason": "fingerprint déjà ingéré",
                    "existing_doc_id": fingerprints[fingerprint],
                })
                continue
            # force=True → supprime les anciens chunks de ce doc_id
            old_doc_id = fingerprints[fingerprint]
            cumulative = [c for c in cumulative if c.get("doc_id") != old_doc_id]
            fingerprints = _indexer.existing_fingerprints(cumulative)
            log.warning("force=True : doc_id %s remplacé (fichier %s)", old_doc_id, path.name)

        # Détermine le doc_id (forcé ou auto-incrémenté depuis le cumul)
        if forced_doc_id:
            doc_id = forced_doc_id
            # Vérif collision dans l'existant cumulé
            if any(c.get("doc_id") == doc_id for c in cumulative):
                if not force:
                    raise ValueError(f"doc_id {doc_id} déjà présent (utiliser --force pour remplacer)")
                cumulative = [c for c in cumulative if c.get("doc_id") != doc_id]
        else:
            doc_id = _indexer.next_doc_id(cumulative)

        chunks = build_chunks_from_file(path, doc_id=doc_id, doc_title=doc_title)
        cumulative.extend(chunks)
        new_chunks.extend(chunks)
        fingerprints[fingerprint] = doc_id

        added.append({
            "file": path.name,
            "doc_id": doc_id,
            "n_chunks": len(chunks),
            "tags": sorted({t for c in chunks for t in c["metadata"].get("tags", [])}),
        })
        log.info("Ingéré %s → %s (%d chunks)", path.name, doc_id, len(chunks))

    if dry_run:
        log.info("DRY-RUN : aucune écriture (ajout=%d, skip=%d)", len(added), len(skipped))
        return {"added": added, "skipped": skipped, "total_chunks_after": len(cumulative), "dry_run": True}

    if added or (force and any(s for s in skipped)):
        _indexer.save_chunks_with_backup(cumulative)
        _indexer.rebuild_index()
    else:
        log.info("Rien à écrire (aucun ajout)")

    return {"added": added, "skipped": skipped, "total_chunks_after": len(cumulative), "dry_run": False}


def list_docs(filter_doc_id: str | None = None) -> list[dict]:
    """Liste les documents groupés par doc_id.

    Retourne : [{"doc_id", "doc_title", "n_chunks", "source_filename",
                 "ingested_at", "tags"}], trié par doc_id croissant.
    """
    chunks = _indexer.load_chunks()
    if filter_doc_id:
        chunks = [c for c in chunks if c.get("doc_id") == filter_doc_id]

    grouped: dict[str, dict] = {}
    for c in chunks:
        did = c.get("doc_id", "?")
        md = c.get("metadata") or {}
        if did not in grouped:
            grouped[did] = {
                "doc_id": did,
                "doc_title": c.get("doc_title", ""),
                "n_chunks": 0,
                "source_filename": md.get("source_filename", ""),
                "ingested_at": md.get("ingested_at", ""),
                "tags": set(),
            }
        grouped[did]["n_chunks"] += 1
        grouped[did]["tags"].update(md.get("tags", []))

    out = []
    for did in sorted(grouped):
        item = grouped[did]
        item["tags"] = sorted(item["tags"])
        out.append(item)
    return out


def get_chunks_for_doc(doc_id: str) -> list[dict]:
    """Retourne tous les chunks d'un doc_id, triés par section_id."""
    chunks = _indexer.load_chunks()
    filtered = [c for c in chunks if c.get("doc_id") == doc_id]
    return sorted(filtered, key=lambda c: c.get("section_id", ""))


def delete_doc(doc_id: str) -> dict:
    """Supprime tous les chunks d'un doc_id. Retourne {"deleted": n, "remaining": n}.

    Raise ValueError si doc_id absent.
    """
    if not _DOC_ID_PATTERN.match(doc_id):
        raise ValueError(f"doc_id invalide : {doc_id} (regex ^D\\d{{2,3}}$)")
    chunks = _indexer.load_chunks()
    before = len(chunks)
    remaining = [c for c in chunks if c.get("doc_id") != doc_id]
    deleted = before - len(remaining)
    if deleted == 0:
        raise ValueError(f"doc_id {doc_id} absent du corpus")
    _indexer.save_chunks_with_backup(remaining)
    _indexer.rebuild_index()
    log.info("Supprimé %d chunks de %s, reste %d", deleted, doc_id, len(remaining))
    return {"deleted": deleted, "remaining": len(remaining), "doc_id": doc_id}


# ─────────────────────────────────────────────────────────────────────────────
# CLI handlers
# ─────────────────────────────────────────────────────────────────────────────

def cmd_add(args) -> int:
    src_dir: Path = args.dir
    if not src_dir.exists() or not src_dir.is_dir():
        log.error("Dossier introuvable : %s", src_dir)
        return 1
    files = sorted([p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_EXT])
    if not files:
        log.error("Aucun PDF/DOCX dans %s", src_dir)
        return 1

    try:
        report = ingest_files(
            files,
            forced_doc_id=args.doc_id,
            doc_title=args.doc_title,
            force=args.force,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_list(args) -> int:
    docs = list_docs(filter_doc_id=args.doc_id)
    if args.format == "json":
        print(json.dumps(docs, ensure_ascii=False, indent=2))
    else:
        if not docs:
            print("(aucun document)")
            return 0
        # Affichage tabulaire simple
        widths = {"doc_id": 5, "n_chunks": 8, "title": 50, "source": 30, "tags": 30}
        header = f"{'doc_id':<{widths['doc_id']}} {'chunks':>{widths['n_chunks']}}  {'doc_title':<{widths['title']}}  {'source':<{widths['source']}}  tags"
        print(header)
        print("-" * len(header))
        for d in docs:
            title = (d["doc_title"][:widths["title"]-1] + "…") if len(d["doc_title"]) > widths["title"] else d["doc_title"]
            src = (d["source_filename"][:widths["source"]-1] + "…") if len(d["source_filename"]) > widths["source"] else d["source_filename"]
            print(f"{d['doc_id']:<{widths['doc_id']}} {d['n_chunks']:>{widths['n_chunks']}}  {title:<{widths['title']}}  {src:<{widths['source']}}  {','.join(d['tags'])}")
        print()
        print(f"Total: {len(docs)} documents, {sum(d['n_chunks'] for d in docs)} chunks")
    return 0


def cmd_delete(args) -> int:
    if not args.confirm:
        ans = input(f"Supprimer définitivement {args.doc_id} et tous ses chunks ? [y/N] ").strip().lower()
        if ans != "y":
            print("Annulé.")
            return 0
    try:
        rep = delete_doc(args.doc_id)
    except ValueError as exc:
        log.error("%s", exc)
        return 1
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


def cmd_rebuild(args) -> int:
    _indexer.rebuild_index()
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# argparse
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manage_doctrine",
        description="Gestion du corpus doctrinal RAG (FAISS).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Ingère les PDF/DOCX d'un dossier")
    p_add.add_argument("--dir", type=Path, required=True, help="Dossier source (PDF + DOCX)")
    p_add.add_argument("--doc-id", default=None, help="Force doc_id (regex ^D\\d{2,3}$), interdit si >1 fichier")
    p_add.add_argument("--doc-title", default=None, help="Titre du document (défaut: nom fichier)")
    p_add.add_argument("--force", action="store_true", help="Remplace si fingerprint déjà ingéré")
    p_add.add_argument("--dry-run", action="store_true", help="Affiche le plan sans rien écrire")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="Liste les documents indexés")
    p_list.add_argument("--doc-id", default=None, help="Filtre par doc_id")
    p_list.add_argument("--format", choices=["table", "json"], default="table")
    p_list.set_defaults(func=cmd_list)

    p_del = sub.add_parser("delete", help="Supprime un doc_id et tous ses chunks")
    p_del.add_argument("--doc-id", required=True, help="doc_id à supprimer (ex: D13)")
    p_del.add_argument("--confirm", action="store_true", help="Bypass prompt confirmation")
    p_del.set_defaults(func=cmd_delete)

    p_reb = sub.add_parser("rebuild", help="Reconstruit l'index FAISS depuis chunks_enriched.json")
    p_reb.set_defaults(func=cmd_rebuild)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
