"""
TOOL CONTRACT — build_pdf.search_methodology
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : build_pdf.search_methodology
domain        : tous
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-01

DESCRIPTION
-----------
Recherche dans la documentation méthodologique actuarielle indexée
(base_de_connaissance/methodologie/chromadb/). Couvre : lissage,
validation statistique, SMR, tests chi-deux, Whittaker-Henderson,
splines, intervalles de confiance, Kimeldorf-Jones, etc.
Distinct de search_exemplars : ici on cherche de la théorie et des
formules, pas des exemples de rapports rédigés.

WHEN TO USE
-----------
Quand l'agent doit justifier un choix méthodologique dans le rapport :
- Choix du paramètre lambda pour Whittaker-Henderson
- Interprétation d'un test chi-deux ou d'un SMR
- Explication d'une méthode de lissage dans la section §2 méthode
- Formulation des hypothèses d'un test statistique

WHEN NOT TO USE
---------------
Ne pas utiliser pour chercher des formulations de rapport rédigé —
utiliser search_exemplars. Ne pas utiliser pour des calculs.

PREREQUISITES
-------------
required_tools: []
required_data_store_keys: []

INPUTS
------
params:
  query:
    type    : string
    note    : >
      Question technique sur une méthode actuarielle.
      Exemples :
        "quel critère pour choisir lambda dans Whittaker-Henderson ?"
        "comment interpréter un SMR de 0.87 avec IC 95% ?"
        "hypothèses du test des changements de signe"
  n_results:
    type    : int
    default : 3
    note    : Nombre de chunks retournés (1-8).

OUTPUTS
-------
data_store_keys_written: []
return_payload:
  chunks     : list de dicts {chunk_id, contenu, score, source}
  n_returned : int
  warning    : string | null

QUALITY GATES
-------------
BLOCKING:
  - query absente → retourner erreur
NON-BLOCKING:
  - score max < 0.4 → warning "Documentation insuffisante sur ce sujet"
  - corpus vide après tentative d'ingestion auto → warning + chunks vide

ERROR HANDLING
--------------
error: "Collection méthodologie non disponible"
  → cause  : ChromaDB absent ou chemin invalide
  → action : retourner chunks vide + warning, ne pas bloquer

AGENT GUIDANCE
--------------
reasoning_hint: >
  Appeler avant de rédiger §2 méthode pour appuyer la justification
  du choix de lissage sur une référence technique.
  Les chunks retournés contiennent des extraits du cours ISFA de
  Planchet sur les méthodes de lissage (Seance6.pdf).
exemplar_query: >
  non applicable

CATALOGUE METADATA
------------------
display_name      : Recherche documentation méthodologique
short_description : Recherche dans la documentation technique actuarielle.
domain            : tous
capability_group  : rag
depends_on        : []
required_by       : []
client_visible    : false
"""
from __future__ import annotations

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CHROMA_PATH  = _PROJECT_ROOT / "base_de_connaissance" / "methodologie" / "chromadb"
_COLLECTION   = "documentation_actuarielle"
_MAX_CONTENT  = 800

# Chemins candidats pour le JSON de documentation
_JSON_CANDIDATES = [
    _PROJECT_ROOT / "Knowledge Base" / "04_smoothing.json",
    _PROJECT_ROOT / "lissage_chunks_complet.json",
    _PROJECT_ROOT / "Knowledge Base" / "lissage_chunks_complet.json",
]


def run(data: dict, params: dict | None = None) -> dict:
    params = params or {}

    query = params.get("query", "")
    if not query or not query.strip():
        return {"erreur": "Le paramètre 'query' est requis."}

    n_results = int(params.get("n_results", 3))
    n_results = max(1, min(n_results, 8))

    try:
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        _CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
        ef     = DefaultEmbeddingFunction()

        try:
            collection = client.get_collection(name=_COLLECTION, embedding_function=ef)
        except Exception:
            collection = client.get_or_create_collection(name=_COLLECTION, embedding_function=ef)

        # Auto-ingestion si collection vide
        if collection.count() == 0:
            _auto_ingest(collection)

        count = collection.count()
        if count == 0:
            return {
                "chunks":     [],
                "n_returned": 0,
                "warning": (
                    "Documentation méthodologique non disponible. "
                    "Exécuter base_de_connaissance/ingest_methodology.py pour initialiser."
                ),
            }

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )

    except ImportError:
        return {
            "chunks":     [],
            "n_returned": 0,
            "warning": "chromadb non installé. Installer avec : pip install chromadb",
        }
    except Exception as exc:
        return {
            "chunks":     [],
            "n_returned": 0,
            "warning": f"Collection méthodologie non disponible ({exc}). Continuer sans référence.",
        }

    docs      = results.get("documents",  [[]])[0]
    metadatas = results.get("metadatas",  [[]])[0]
    distances = results.get("distances",  [[]])[0]
    ids       = results.get("ids",        [[]])[0]

    chunks = []
    max_score = 0.0
    for doc, meta, dist, cid in zip(docs, metadatas, distances, ids):
        score = float(1.0 - dist)
        max_score = max(max_score, score)
        content_trunc = (doc or "")[:_MAX_CONTENT]
        if len(doc or "") > _MAX_CONTENT:
            content_trunc += " […]"
        chunks.append({
            "chunk_id": cid,
            "contenu":  content_trunc,
            "score":    round(score, 4),
            "source":   (meta or {}).get("source", "inconnu"),
        })

    chunks.sort(key=lambda c: c["score"], reverse=True)

    warning = None
    if chunks and max_score < 0.4:
        warning = f"Documentation insuffisante sur ce sujet (score max: {max_score:.3f})."

    return {
        "chunks":     chunks,
        "n_returned": len(chunks),
        "warning":    warning,
    }


def _auto_ingest(collection) -> None:
    """Tente d'ingérer automatiquement la documentation depuis les fichiers JSON candidats."""
    for json_path in _JSON_CANDIDATES:
        if not json_path.exists():
            continue
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(raw, list) or not raw:
                continue
            ids, docs, metas = [], [], []
            for item in raw:
                chunk_id = item.get("id") or item.get("chunk_id") or f"chunk_{len(ids)}"
                contenu  = item.get("contenu") or item.get("content") or ""
                if not contenu:
                    continue
                meta = {
                    "source":  item.get("source", json_path.name),
                    "section": item.get("section", "inconnu"),
                    "titre":   item.get("titre", ""),
                    "type":    item.get("type", "methodologie"),
                }
                ids.append(chunk_id)
                docs.append(contenu)
                metas.append(meta)

            if ids:
                collection.add(ids=ids, documents=docs, metadatas=metas)
            return  # succès sur le premier fichier trouvé
        except Exception:
            continue
