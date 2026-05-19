"""
agents.rag.describe_capabilities

Self-describe du RAGAgent — appelé au démarrage par capability_registry.

Le RAGAgent répond aux questions doctrinales actuarielles (méthodes,
réglementaire FR). N'exécute pas de calcul. Mémoire conversationnelle
3-niveaux pour résoudre les anaphores multi-tour.
"""
from __future__ import annotations

from pathlib import Path


_META_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "knowledge_base" / "rag_doctrine" / "index" / "meta.json"
)


def _corpus_stats() -> dict:
    """Lit meta.json du corpus FAISS pour exposer les stats à jour."""
    if not _META_PATH.exists():
        return {"n_chunks": 0, "n_docs": 0, "embedder": "indisponible"}
    import json
    try:
        with _META_PATH.open(encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return {"n_chunks": 0, "n_docs": 0, "embedder": "erreur de chargement"}
    chunks = meta.get("chunks", [])
    doc_ids = {c.get("doc_id") for c in chunks if c.get("doc_id")}
    return {
        "n_chunks": len(chunks),
        "n_docs":   len(doc_ids),
        "embedder": meta.get("embedder", "inconnu"),
        "doc_ids":  sorted(doc_ids),
    }


def describe() -> dict:
    """Retourne le descriptif des capacités de retrieval du RAGAgent."""
    stats = _corpus_stats()
    return {
        "agent":   "rag",
        "display": "RAGAgent — Questions doctrinales",
        "purpose": (
            "Répond aux questions méthodologiques et réglementaires "
            "actuarielles à partir du corpus de doctrine française. "
            "N'exécute aucun calcul. Conserve l'historique conversationnel "
            "sur 3 niveaux pour résoudre les anaphores multi-tour."
        ),
        "tools": [
            {
                "name":        "conversation.search_doctrine",
                "display":     "Retrieval hybride doctrine",
                "description": "FAISS dense + BM25 sparse + RRF fusion sur le corpus indexé.",
            },
        ],
        "corpus_stats":   stats,
        "topics_covered": [
            "Préparation données (D01)",
            "Estimateurs taux bruts : Kaplan-Meier, Nelson-Aalen (D02)",
            "Lissage : Whittaker-Henderson, méthodes paramétriques (D03)",
            "Validation : chi-2, SMR, runs (D04)",
            "Fermeture grands âges : Coale-Kisker, Denuit-Goderniaux (D05)",
            "Tables prospectives : Lee-Carter, Brouhns-Denuit-Vermunt, Cairns-Blake-Dowd (D06)",
            "Cadre réglementaire FR : A132-18, BCAC, arrêtés (D07)",
            "Tables réglementaires : TH/TF 00-02, TGH/TGF 05, TPRV 93, TD/TV 88-90 (D08)",
            "Certification IA et prudence (D09)",
            "Solvabilité 2 et marges (D10)",
            "Landscape international (D11)",
            "Annexes méthodologiques (D12)",
        ],
        "memory_levels": [
            "Niveau 1 (buffer) : 4 derniers tours verbatim",
            "Niveau 2 (summary) : résumé structuré incrémental au-delà de 10 tours",
            "Niveau 3 (vectorstore) : FAISS MiniLM-384 de tous les Q/A passés",
        ],
        "safety_layers": [
            "Sanitize input (truncate + control chars)",
            "Détection jailbreak (16 regex FR+EN)",
            "Filtre scope lexical (488 termes corpus)",
            "Prompt hardening anti-injection",
            "Citation check (rejet si [Dxx.yy] manquant)",
            "Memory hygiene (neutralisation marqueurs structurels)",
        ],
        "inputs":  ["question utilisateur en langage naturel"],
        "outputs": [
            "réponse rédigée 3-6 phrases avec citations [Dxx.yy] inline",
            "section Sources listant les chunks utilisés",
        ],
    }
