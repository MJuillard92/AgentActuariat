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
    note    : ⚠️ Laisser FALSE par défaut. Activer rerank charge un modèle
              cross-encoder de 600 Mo (cold start +5-10s) — pas justifié
              pour 99% des questions. Réservé aux cas où FAISS+BM25 ne
              retournent que des résultats hors-sujet (rare sur 142 chunks).

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
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Force HuggingFace en mode offline si les modèles sont déjà en cache.
# Sans ça, sentence-transformers fait un HEAD HTTP à chaque chargement
# pour check les updates → 2-5s de latence + risque "Connection reset"
# observé en prod (HF rate-limit anonyme).
_HF_HUB_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
if _HF_HUB_CACHE.exists() and any(_HF_HUB_CACHE.iterdir()):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Chemins du corpus + index FAISS (immutable, embarqué avec le projet)
_DOCTRINE_DIR  = Path(__file__).resolve().parent.parent.parent / "knowledge_base" / "rag_doctrine"
_INDEX_PATH    = _DOCTRINE_DIR / "index" / "faiss.bin"
_META_PATH     = _DOCTRINE_DIR / "index" / "meta.json"

# Cache du retriever (chargement modèle ~1-10s selon embedder).
# Réutilisé pour tous les appels de la session pour éviter le cold start
# à chaque question.
_RETRIEVER_CACHE: dict[str, Any] = {}


def _get_retriever():
    """Lazy load du HybridRetriever. L'embedder est auto-détecté depuis
    le meta.json — garantit la cohérence entre l'index construit et les
    requêtes (mismatch dim = recall nul)."""
    if "instance" in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE["instance"]
    if not _INDEX_PATH.exists() or not _META_PATH.exists():
        raise FileNotFoundError(
            f"Index FAISS doctrine absent ({_INDEX_PATH}). "
            "Lancer : python knowledge_base/rag_doctrine/ingest_doctrine.py"
        )
    # Lire l'embedder utilisé pour construire l'index
    import json as _json
    with _META_PATH.open(encoding="utf-8") as f:
        meta = _json.load(f)
    embedder_name = meta.get("embedder", "bge-m3")
    log.info("Chargement du retriever doctrine (embedder=%s) — cache RAM …",
             embedder_name)
    from tools.conversation._retriever._pack_retriever import HybridRetriever
    r = HybridRetriever.from_paths(_INDEX_PATH, _META_PATH, embedder=embedder_name)
    _RETRIEVER_CACHE["instance"] = r
    return r


def warmup() -> bool:
    """Pré-charge le retriever (modèle embedder + index FAISS) pour
    éliminer le cold start au 1er appel utilisateur. À appeler une fois
    au démarrage de l'application. Retourne True si OK, False sinon.
    No-op si déjà chargé."""
    try:
        _get_retriever()
        return True
    except Exception as exc:
        log.warning("warmup search_doctrine échoué : %s", exc)
        return False


# Cache des métadonnées brutes (chunks list). Évite de relire meta.json
# à chaque lookup. Indépendant du retriever (pas d'embedder).
_META_CACHE: dict[str, Any] = {}


def _load_meta_chunks() -> list[dict]:
    """Charge la liste des chunks depuis meta.json (cached)."""
    if "chunks" in _META_CACHE:
        return _META_CACHE["chunks"]
    if not _META_PATH.exists():
        raise FileNotFoundError(f"Index meta.json doctrine absent ({_META_PATH}).")
    import json as _json
    with _META_PATH.open(encoding="utf-8") as f:
        meta = _json.load(f)
    _META_CACHE["chunks"] = meta.get("chunks") or []
    return _META_CACHE["chunks"]


def get_chunks_by_doc_id(doc_id: str, max_chunks: int = 3) -> list[dict]:
    """Lookup direct des chunks d'un document (pas de query sémantique).

    Pour injection ciblée dans un prompt de rédaction : on sait qu'on
    veut citer la fiche D03_lissage section smoothing, donc on lit ses
    N premiers chunks (ordre du document) directement.

    Args:
        doc_id : code court du document (ex. "D03"). Accepte aussi les
            formes étendues `D03_lissage` ou `D03.01` (préfixe matché).
        max_chunks : nombre max de chunks à retourner. Par défaut 3.

    Returns:
        list[dict] avec : chunk_id, doc_id, section_id, section_title,
        section_path, text, tags (vide si non défini). Liste vide si
        doc_id inconnu.

    Plan refonte PDF 2026-06-03 (Axe B.2 — RAG doctrine direct).
    """
    if not doc_id:
        return []
    try:
        chunks = _load_meta_chunks()
    except FileNotFoundError as exc:
        log.warning("get_chunks_by_doc_id : %s", exc)
        return []

    # Normalisation : « D03_lissage » ou « D03.01 » → « D03 ».
    target = str(doc_id).split("_")[0].split(".")[0].strip().upper()

    selected = [c for c in chunks if str(c.get("doc_id", "")).upper() == target]
    selected.sort(key=lambda c: str(c.get("section_id", "")))
    out = []
    for c in selected[: max(0, int(max_chunks))]:
        md = c.get("metadata") or {}
        out.append({
            "chunk_id":      c.get("chunk_id"),
            "doc_id":        c.get("doc_id"),
            "section_id":    c.get("section_id"),
            "section_title": c.get("section_title"),
            "section_path":  c.get("section_path"),
            "text":          c.get("text", ""),
            "tags":          md.get("tags", []),
        })
    return out


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
