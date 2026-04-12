"""
TOOL CONTRACT — build_pdf.search_exemplars
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : build_pdf.search_exemplars
domain        : tous
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-01

DESCRIPTION
-----------
Recherche sémantique dans le corpus d'exemplaires de rapports actuariels
(base_de_connaissance/exemplaires/chromadb/). Retourne les N chunks les
plus pertinents avec leurs métadonnées. Utilisé par l'agent avant de
rédiger chaque section du rapport pour s'appuyer sur des précédents
professionnels.

WHEN TO USE
-----------
Avant de rédiger le commentary dans certification_report.
Obligatoirement aux trois points de jugement :
- §2 méthode    : query sur le choix de lissage et sa justification
- §3 résultats  : query sur l'interprétation SMR et des anomalies
- §5 conclusion : query sur la formulation des recommandations d'usage

WHEN NOT TO USE
---------------
Ne pas appeler si le corpus est vide — retourner warning sans erreur.
Ne pas appeler pour des questions méthodologiques techniques —
utiliser build_pdf.search_methodology à la place.
Ne pas appeler pour des calculs — utiliser les tools builder.

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
      Question en langage naturel décrivant ce que l'agent cherche.
      Formuler comme une question métier, pas comme un mot-clé.
      Exemples :
        "comment formuler la prudence d'une table de mortalité temporaire décès"
        "justification choix whittaker-henderson portefeuille 50 000 vies"
        "formulation conclusion recommandation usage provisionnement"
        "comment présenter les abattements réglementaires"
  n_results:
    type    : int
    default : 3
    note    : 3 suffit pour la plupart des usages. Augmenter à 5 si la query couvre plusieurs angles.
  filters:
    type    : dict
    default : {}
    note    : >
      Filtres optionnels sur les métadonnées ChromaDB.
      Exemples :
        {"type_rapport": "construction"}
        {"produit": "temporaire_deces"}
        {"qualite_exemplaire": "high"}
        {"methode_lissage": "whittaker"}

OUTPUTS
-------
data_store_keys_written: []
return_payload:
  chunks     : list de dicts {chunk_id, section, contenu, score, metadata}
  n_returned : int
  query_used : string
  warning    : string | null

QUALITY GATES
-------------
BLOCKING:
  - query absente ou vide → retourner erreur "query est requise"
NON-BLOCKING:
  - score max < 0.5 → warning "Aucun exemplaire très pertinent trouvé (score max: {score}). Rédiger avec prudence sans précédent direct."
  - corpus vide → warning + chunks vide, continuer sans bloquer

ERROR HANDLING
--------------
error: "ChromaDB non disponible"
  → cause  : collection non initialisée ou chemin introuvable
  → action : retourner {"chunks": [], "n_returned": 0, "warning": "Corpus exemplaires non disponible — continuer sans précédent."}
             Ne jamais bloquer l'agent sur une erreur RAG.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Appeler ce tool une fois par paragraphe majeur du commentary.
  Lire le champ "contenu" des chunks retournés pour extraire
  les formulations et la structure — ne pas recopier mot pour mot.
  Si score < 0.5 sur tous les chunks, signaler dans le commentary
  qu'aucun précédent similaire n'est disponible.
exemplar_query: >
  non applicable

CATALOGUE METADATA
------------------
display_name      : Recherche exemplaires RAG
short_description : Recherche sémantique dans le corpus de rapports exemplaires.
domain            : tous
capability_group  : rag
depends_on        : []
required_by       : [build_pdf.certification_report]
client_visible    : false
"""
from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CHROMA_PATH  = _PROJECT_ROOT / "base_de_connaissance" / "exemplaires" / "chromadb"
_COLLECTION   = "exemplaires_actuariels"
_MAX_CONTENT  = 800  # caractères max par chunk retourné


def run(data: dict, params: dict | None = None) -> dict:
    params = params or {}

    query = params.get("query", "")
    if not query or not query.strip():
        return {"erreur": "Le paramètre 'query' est requis et ne peut pas être vide."}

    n_results = int(params.get("n_results", 3))
    n_results = max(1, min(n_results, 8))
    filters   = params.get("filters", {}) or {}

    try:
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        _CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
        ef     = DefaultEmbeddingFunction()

        # Obtenir ou créer la collection
        try:
            collection = client.get_collection(
                name=_COLLECTION,
                embedding_function=ef,
            )
        except Exception:
            collection = client.get_or_create_collection(
                name=_COLLECTION,
                embedding_function=ef,
            )

        # Vérifier si la collection est vide
        count = collection.count()
        if count == 0:
            return {
                "chunks":     [],
                "n_returned": 0,
                "query_used": query,
                "warning": (
                    "Corpus exemplaires vide. Aucun rapport n'a encore été ingéré. "
                    "Utiliser base_de_connaissance/chunk_report.py pour ajouter des exemplaires."
                ),
            }

        # Construire les where filters pour ChromaDB
        where = _build_where(filters)

        # Requête
        query_kwargs: dict = {
            "query_texts": [query],
            "n_results":   min(n_results, count),
            "include":     ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = collection.query(**query_kwargs)

    except ImportError:
        return {
            "chunks":     [],
            "n_returned": 0,
            "query_used": query,
            "warning": "chromadb non installé. Installer avec : pip install chromadb",
        }
    except Exception as exc:
        return {
            "chunks":     [],
            "n_returned": 0,
            "query_used": query,
            "warning": f"Corpus exemplaires non disponible — continuer sans précédent. ({exc})",
        }

    # Parser les résultats
    chunks = []
    docs       = results.get("documents",  [[]])[0]
    metadatas  = results.get("metadatas",  [[]])[0]
    distances  = results.get("distances",  [[]])[0]
    ids        = results.get("ids",        [[]])[0]

    max_score = 0.0
    for doc, meta, dist, cid in zip(docs, metadatas, distances, ids):
        score = float(1.0 - dist)
        max_score = max(max_score, score)
        content_trunc = (doc or "")[:_MAX_CONTENT]
        if len(doc or "") > _MAX_CONTENT:
            content_trunc += " […]"
        chunks.append({
            "chunk_id": cid,
            "section":  (meta or {}).get("section", "inconnu"),
            "contenu":  content_trunc,
            "score":    round(score, 4),
            "metadata": meta or {},
        })

    # Trier par score décroissant
    chunks.sort(key=lambda c: c["score"], reverse=True)

    warning = None
    if chunks and max_score < 0.5:
        warning = (
            f"Aucun exemplaire très pertinent trouvé (score max: {max_score:.3f}). "
            "Rédiger avec prudence sans précédent direct."
        )

    return {
        "chunks":     chunks,
        "n_returned": len(chunks),
        "query_used": query,
        "warning":    warning,
    }


def _build_where(filters: dict) -> dict | None:
    """Convertit un dict de filtres en clause ChromaDB where."""
    if not filters:
        return None
    if len(filters) == 1:
        k, v = next(iter(filters.items()))
        return {k: {"$eq": v}}
    # Plusieurs filtres → $and
    return {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}
