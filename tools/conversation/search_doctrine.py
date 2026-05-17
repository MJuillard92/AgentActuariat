"""
TOOL CONTRACT — conversation.search_doctrine
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : conversation.search_doctrine
domain        : conversation
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-05-17

DESCRIPTION
-----------
Recherche dans la doctrine actuarielle française : 142 chunks indexés
(FAISS dense + BM25 sparse + RRF fusion + reranker optionnel) couvrant
préparation données, estimateurs taux bruts, lissage, validation
(chi², SMR, runs), fermeture grands âges, tables prospectives (Lee-Carter,
Brouhns-Denuit-Vermunt, Cairns-Blake-Dowd), cadre réglementaire FR
(A132-18, BCAC, arrêtés), tables réglementaires (TH/TF 00-02, TGH/TGF 05,
TPRV 93, TD/TV 88-90), certification IA, prudence et marges,
Solvabilité 2, landscape international.

WHEN TO USE
-----------
Mode conversation, l'utilisateur pose une QUESTION méthodologique ou
réglementaire :
  - "C'est quoi le lissage Whittaker-Henderson ?"
  - "Comment fonctionne le test du chi-2 sur les tables ?"
  - "Quelle est l'obligation A132-18 ?"
  - "Différence entre table périodique et prospective ?"
  - "Comment calibrer le paramètre h ?"
Le tool retourne les chunks pertinents avec sources (doc_id, section_id) ;
le LLM doit ensuite reformuler en langage naturel et CITER ces sources.

WHEN NOT TO USE
---------------
Pour les calculs actuariels (utiliser le pipeline Builder).
Pour l'inspection du fichier user (utiliser data_inspect / eval_pandas).

INPUTS
------
params:
  query:
    type    : string
    note    : Question utilisateur reformulée pour la recherche.
  k:
    type    : int
    default : 5
    note    : Nombre de chunks à retourner.
  filters:
    type    : dict | null
    default : null
    note    : Filtres métadonnées optionnels :
              {"doc_id": "D03"} → uniquement section lissage
              {"tags": ["validation"]} → uniquement tests/validation
              {"regulatory": true} → uniquement chunks citant Code des assurances
              {"tables_referenced": ["TH_00_02"]} → contient cette table
  rerank:
    type    : bool
    default : false
    note    : Si true, applique le reranker cross-encoder (latence +500ms-1s,
              précision +5-10 pts recall@5). Réservé aux questions ambiguës.

OUTPUTS
-------
return_payload:
  results : list[dict] — {chunk_id, doc_id, section_id, section_title,
                          section_path, text, score, tags, regulatory,
                          tables_referenced}
  n_returned : int
  query_used : str

AGENT GUIDANCE
--------------
reasoning_hint: >
  TOUJOURS reformuler la réponse en langage naturel — ne pas balancer le
  texte brut du chunk. CITER au minimum le doc_id + section_title
  (ex: "selon D03.02 - Whittaker-Henderson 1D : ..."). Ne JAMAIS inventer
  des références : si un chunk ne contient pas l'info, le dire honnêtement.

CATALOGUE METADATA
------------------
display_name      : Recherche doctrine actuarielle française
short_description : RAG hybride sur 142 chunks (méthodes + réglementaire FR).
domain            : conversation
capability_group  : data_exploration
client_visible    : true
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Chemins du corpus + index FAISS (immutable, embarqué avec le projet)
_DOCTRINE_DIR  = Path(__file__).resolve().parent.parent.parent / "knowledge_base" / "rag_doctrine"
_INDEX_PATH    = _DOCTRINE_DIR / "index" / "faiss.bin"
_META_PATH     = _DOCTRINE_DIR / "index" / "meta.json"
_EMBEDDER      = "bge-m3"   # cohérent avec l'index construit

# Cache du retriever (chargement modèle ~5-10s, on évite de le faire
# à chaque appel — réutilisé pour tous les appels de la session)
_RETRIEVER_CACHE: dict[str, Any] = {}


def _get_retriever():
    """Lazy load du HybridRetriever. Modèle bge-m3 chargé en RAM au 1er appel."""
    if "instance" in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE["instance"]
    if not _INDEX_PATH.exists() or not _META_PATH.exists():
        raise FileNotFoundError(
            f"Index FAISS doctrine absent ({_INDEX_PATH}). "
            "Lancer : python knowledge_base/rag_doctrine/ingest_doctrine.py"
        )
    from tools.conversation._retriever._pack_retriever import HybridRetriever
    log.info("Chargement du retriever doctrine (bge-m3) — cache RAM …")
    r = HybridRetriever.from_paths(_INDEX_PATH, _META_PATH, embedder=_EMBEDDER)
    _RETRIEVER_CACHE["instance"] = r
    return r


def run(df=None, params: dict | None = None) -> dict:
    """Point d'entrée tool. `df` est ignoré (le retriever a son propre corpus)."""
    params = params or {}
    query = params.get("query", "")
    if not query or not isinstance(query, str):
        return {"erreur": "param 'query' (string) requis"}

    k = int(params.get("k", 5))
    rerank = bool(params.get("rerank", False))
    filters = params.get("filters") or None

    try:
        retriever = _get_retriever()
    except FileNotFoundError as exc:
        return {"erreur": str(exc)}
    except Exception as exc:
        return {"erreur": f"Initialisation retriever échouée : {exc}"}

    try:
        hits = retriever.retrieve(query, k=k, filters=filters, rerank=rerank)
    except Exception as exc:
        return {"erreur": f"Recherche échouée : {exc}"}

    results = []
    for h in hits:
        md = h.metadata or {}
        results.append({
            "chunk_id":          h.chunk_id,
            "doc_id":            h.doc_id,
            "section_id":        h.section_id,
            "section_title":     h.section_title,
            "score":             round(float(h.score), 4),
            "text":              h.text,
            "tags":              md.get("tags", []),
            "regulatory":        bool(md.get("regulatory", False)),
            "tables_referenced": md.get("tables_referenced", []),
        })

    return {
        "query_used":  query,
        "n_returned":  len(results),
        "results":     results,
    }
