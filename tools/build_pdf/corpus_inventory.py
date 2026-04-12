"""
TOOL CONTRACT — build_pdf.corpus_inventory
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : build_pdf.corpus_inventory
domain        : tous
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-01

DESCRIPTION
-----------
Inventorie le corpus d'exemplaires actuariels disponible dans
(base_de_connaissance/exemplaires/chromadb/). Retourne la liste
des rapports ingérés avec leurs métadonnées clés, dédupliquée
par rapport_id. Utile pour vérifier la richesse du corpus avant
de lancer une recherche sémantique ou avant d'ingérer un nouveau
rapport.

WHEN TO USE
-----------
- En début de session pour vérifier que le corpus contient des
  exemplaires pertinents (produit similaire, méthode similaire).
- Avant d'appeler search_exemplars pour choisir des filtres.
- Pour diagnostiquer un corpus vide ou insuffisant.
- Sur demande explicite du client ("quels exemples as-tu ?").

WHEN NOT TO USE
---------------
Ne pas appeler à chaque invocation — une fois par session suffit.
Ne pas utiliser comme substitut à search_exemplars pour trouver
des formulations.

PREREQUISITES
-------------
required_tools: []
required_data_store_keys: []

INPUTS
------
params:
  sort_by:
    type    : string
    default : "qualite_exemplaire"
    note    : Clé de tri primaire sur les métadonnées. Options :
              "qualite_exemplaire" (desc), "date" (desc),
              "type_rapport", "produit".
  limit:
    type    : int
    default : 20
    note    : Nombre maximum de rapports retournés dans la liste.

OUTPUTS
-------
data_store_keys_written: []
return_payload:
  rapports      : list de dicts {rapport_id, type_rapport, produit,
                  methode_lissage, periode_observation,
                  qualite_exemplaire, n_chunks}
  n_rapports    : int   — nombre de rapports distincts
  n_chunks_total: int   — nombre total de chunks dans la collection
  warning       : string | null

QUALITY GATES
-------------
BLOCKING: aucun
NON-BLOCKING:
  - corpus vide → warning "Corpus vide. Utiliser chunk_report.py pour ingérer des exemplaires."
  - n_rapports == 0 après déduplication → warning + rapports vide

ERROR HANDLING
--------------
error: "Collection exemplaires non disponible"
  → cause  : ChromaDB absent ou chemin invalide
  → action : retourner rapports vide + warning, ne pas bloquer

AGENT GUIDANCE
--------------
reasoning_hint: >
  Appeler une fois en début de session si le client demande
  "quels types de rapports as-tu en exemple ?".
  Les champs qualite_exemplaire et methode_lissage permettent
  de choisir les filtres les plus pertinents pour search_exemplars.
exemplar_query: >
  non applicable

CATALOGUE METADATA
------------------
display_name      : Inventaire corpus exemplaires
short_description : Liste les rapports exemplaires disponibles dans le corpus RAG.
domain            : tous
capability_group  : rag
depends_on        : []
required_by       : []
client_visible    : false
"""
from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CHROMA_PATH  = _PROJECT_ROOT / "base_de_connaissance" / "exemplaires" / "chromadb"
_COLLECTION   = "exemplaires_actuariels"

# Clés de métadonnées reconnues pour chaque rapport
_META_KEYS = [
    "rapport_id",
    "type_rapport",
    "produit",
    "methode_lissage",
    "periode_observation",
    "qualite_exemplaire",
]

# Ordre de qualité pour le tri
_QUALITE_ORDER = {"high": 0, "medium": 1, "low": 2, "": 3}


def run(data: dict, params: dict | None = None) -> dict:
    params    = params or {}
    sort_by   = params.get("sort_by", "qualite_exemplaire")
    limit     = int(params.get("limit", 20))
    limit     = max(1, min(limit, 200))

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

        n_chunks_total = collection.count()
        if n_chunks_total == 0:
            return {
                "rapports":       [],
                "n_rapports":     0,
                "n_chunks_total": 0,
                "warning": (
                    "Corpus exemplaires vide. "
                    "Utiliser base_de_connaissance/chunk_report.py pour ingérer des exemplaires."
                ),
            }

        # Récupérer tous les chunks avec leurs métadonnées (sans documents pour économiser la mémoire)
        result = collection.get(include=["metadatas"])
        ids        = result.get("ids", [])
        metadatas  = result.get("metadatas", [])

    except ImportError:
        return {
            "rapports":       [],
            "n_rapports":     0,
            "n_chunks_total": 0,
            "warning": "chromadb non installé. Installer avec : pip install chromadb",
        }
    except Exception as exc:
        return {
            "rapports":       [],
            "n_rapports":     0,
            "n_chunks_total": 0,
            "warning": f"Collection exemplaires non disponible ({exc}).",
        }

    # Dédupliquer par rapport_id, compter les chunks par rapport
    rapport_map: dict[str, dict] = {}
    for chunk_id, meta in zip(ids, metadatas):
        meta = meta or {}
        rid  = meta.get("rapport_id") or meta.get("source") or chunk_id
        if rid not in rapport_map:
            rapport_map[rid] = {
                "rapport_id":          rid,
                "type_rapport":        meta.get("type_rapport", "inconnu"),
                "produit":             meta.get("produit", "inconnu"),
                "methode_lissage":     meta.get("methode_lissage", "inconnu"),
                "periode_observation": meta.get("periode_observation", ""),
                "qualite_exemplaire":  meta.get("qualite_exemplaire", ""),
                "n_chunks":            1,
            }
        else:
            rapport_map[rid]["n_chunks"] += 1

    rapports = list(rapport_map.values())

    # Tri
    if sort_by == "qualite_exemplaire":
        rapports.sort(key=lambda r: (
            _QUALITE_ORDER.get(r["qualite_exemplaire"], 3),
        ))
    elif sort_by == "date":
        rapports.sort(key=lambda r: r.get("periode_observation", ""), reverse=True)
    elif sort_by in ("type_rapport", "produit"):
        rapports.sort(key=lambda r: r.get(sort_by, ""))
    # else: no sort

    rapports = rapports[:limit]

    warning = None
    if not rapports:
        warning = "Aucun rapport trouvé après déduplication."

    return {
        "rapports":       rapports,
        "n_rapports":     len(rapports),
        "n_chunks_total": n_chunks_total,
        "warning":        warning,
    }
