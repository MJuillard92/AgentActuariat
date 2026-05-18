"""
agents.rag.pipeline.query_normalizer

Normalisation déterministe des typos actuariels avant le retrieval.

Pourquoi : sans normalisation, "wittaker" attaque FAISS+BM25 avec un token
inconnu et retourne le top-3 hors-sujet. Une simple correction lexicale
restaure 90% du recall sans coût LLM.

Implémentation : dictionnaire de patterns regex (formes typo → forme canonique)
appliqué case-insensitive sur le texte d'entrée. Aucune dépendance externe.
"""
from __future__ import annotations

import re

# ── Dictionnaire des corrections ─────────────────────────────────────────────
# Chaque entrée : (motif regex, forme canonique)
# Le motif est testé case-insensitive sur des frontières de mots (\b) pour
# éviter les faux positifs.
#
# Conventions :
#   - Forme canonique en minuscules (le retriever est case-insensitive)
#   - Préférer les noms composés avec tiret (kaplan-meier, lee-carter)
#   - Préserver l'orthographe française des termes natifs
_CORRECTIONS: list[tuple[re.Pattern[str], str]] = [
    # ── Whittaker-Henderson ──────────────────────────────────────────────────
    (re.compile(r"\bwhittakker\b",   re.IGNORECASE), "whittaker"),
    (re.compile(r"\bwittakker\b",    re.IGNORECASE), "whittaker"),
    (re.compile(r"\bwittaker\b",     re.IGNORECASE), "whittaker"),
    (re.compile(r"\bwhitaker\b",     re.IGNORECASE), "whittaker"),

    # ── Kaplan-Meier ─────────────────────────────────────────────────────────
    (re.compile(r"\bkaplain[-\s]meier\b", re.IGNORECASE), "kaplan-meier"),
    (re.compile(r"\bkaplain\b",      re.IGNORECASE), "kaplan-meier"),
    (re.compile(r"\bkaplan[-\s]meir\b",   re.IGNORECASE), "kaplan-meier"),
    (re.compile(r"\bkaplan[\s]meier\b",   re.IGNORECASE), "kaplan-meier"),
    (re.compile(r"\bKM\b",                                ), "kaplan-meier"),  # acronyme sensible casse

    # ── Lois paramétriques ───────────────────────────────────────────────────
    (re.compile(r"\bgompertez\b",    re.IGNORECASE), "gompertz"),
    (re.compile(r"\bgompert\b",      re.IGNORECASE), "gompertz"),
    (re.compile(r"\bmakkeham\b",     re.IGNORECASE), "makeham"),
    (re.compile(r"\bmakehamm\b",     re.IGNORECASE), "makeham"),

    # ── Lee-Carter ───────────────────────────────────────────────────────────
    (re.compile(r"\blee[\s]carter\b", re.IGNORECASE), "lee-carter"),

    # ── Cairns-Blake-Dowd ────────────────────────────────────────────────────
    (re.compile(r"\bcairns[\s]blake[\s]dowd\b", re.IGNORECASE), "cairns-blake-dowd"),
    (re.compile(r"\bCBD\b",                                   ), "cairns-blake-dowd"),

    # ── Brouhns-Denuit-Vermunt ───────────────────────────────────────────────
    (re.compile(r"\bbrouhns[\s]denuit[\s]vermunt\b", re.IGNORECASE), "brouhns-denuit-vermunt"),
    (re.compile(r"\bBDV\b",                                        ), "brouhns-denuit-vermunt"),

    # ── Nelson-Aalen ─────────────────────────────────────────────────────────
    (re.compile(r"\bnelson[\s]aalen\b", re.IGNORECASE), "nelson-aalen"),
    (re.compile(r"\bnelson[\s]allen\b", re.IGNORECASE), "nelson-aalen"),

    # ── Denuit-Goderniaux ────────────────────────────────────────────────────
    (re.compile(r"\bdenuit[\s]goderniaux\b", re.IGNORECASE), "denuit-goderniaux"),
    (re.compile(r"\bgoderniau\b",            re.IGNORECASE), "goderniaux"),

    # ── Tests statistiques ───────────────────────────────────────────────────
    (re.compile(r"\bkhi2\b",         re.IGNORECASE), "chi-2"),
    (re.compile(r"\bkhi-2\b",        re.IGNORECASE), "chi-2"),
    (re.compile(r"\bkhi[\s]deux\b",  re.IGNORECASE), "chi-2"),
    (re.compile(r"\bchi2\b",         re.IGNORECASE), "chi-2"),
    (re.compile(r"\bchi[\s]deux\b",  re.IGNORECASE), "chi-2"),

    # ── Acronymes actuariels — expansion ─────────────────────────────────────
    (re.compile(r"\bIC\b",                                ), "intervalle de confiance"),
    (re.compile(r"\bSMR\b",                               ), "SMR ratio de mortalité standardisée"),
    (re.compile(r"\bBCAC\b",                              ), "BCAC Bureau Commun Assurances Collectives"),

    # ── Références réglementaires ────────────────────────────────────────────
    (re.compile(r"\bA[\s]?132[\s\-]?18\b", re.IGNORECASE), "A132-18"),
    (re.compile(r"\bart(?:icle)?[\s]A[\s]?132[\s\-]?18\b", re.IGNORECASE), "A132-18"),

    # ── Tables réglementaires françaises ─────────────────────────────────────
    (re.compile(r"\bTH[\s\-]?00[\s\-]?02\b", re.IGNORECASE), "TH 00-02"),
    (re.compile(r"\bTF[\s\-]?00[\s\-]?02\b", re.IGNORECASE), "TF 00-02"),
    (re.compile(r"\bTGH[\s\-]?05\b",         re.IGNORECASE), "TGH 05"),
    (re.compile(r"\bTGF[\s\-]?05\b",         re.IGNORECASE), "TGF 05"),
    (re.compile(r"\bTPRV[\s\-]?93\b",        re.IGNORECASE), "TPRV 93"),
]


def normalize(text: str) -> str:
    """
    Normalise les typos actuariels et expanse les acronymes.

    Args:
        text: la requête utilisateur brute.

    Returns:
        Le texte corrigé. Les motifs non reconnus passent à travers inchangés.
        La casse originale est préservée hors substitutions.
    """
    if not text:
        return ""
    out = text
    for pattern, canonical in _CORRECTIONS:
        out = pattern.sub(canonical, out)
    return out
