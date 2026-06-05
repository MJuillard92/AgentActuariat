"""
_patterns.py — Registre des regex actuarielles pour enrichissement automatique
des métadonnées de chunks.

Trois familles de patterns :
  - AUTHORS         : auteurs/modèles (Planchet, Lee-Carter, ...)
  - METHOD_TAGS     : tags méthodologiques (mortalite, lissage, prospectif, ...)
                      → un match alimente metadata.tags
  - REGULATORY      : articles de loi/arrêtés/codes → regulatory=True
  - TABLES          : tables réglementaires nommées (TH/TF, TGH/TGF, ...)
                      → metadata.tables_referenced

Patterns extraits empiriquement de chunks_enriched.json (142 chunks doctrinaux).
"""
from __future__ import annotations

import re

# Flags communs : insensible à la casse, multiline
_RE = re.IGNORECASE | re.MULTILINE


# ─────────────────────────────────────────────────────────────────────────────
# Auteurs / modèles nommés
# ─────────────────────────────────────────────────────────────────────────────
AUTHORS: dict[str, re.Pattern] = {
    "auteur_planchet":           re.compile(r"\bPlanchet\b",                                                    _RE),
    "auteur_lee_carter":         re.compile(r"\bLee[- ]Carter\b",                                               _RE),
    "auteur_brouhns":            re.compile(r"\bBrouhns\b",                                                     _RE),
    "auteur_cairns_blake_dowd":  re.compile(r"\bCairns[- ]Blake[- ]Dowd\b|\bCBD\b",                             _RE),
    "auteur_makeham":            re.compile(r"\bMakeham\b",                                                     _RE),
    "auteur_gompertz":           re.compile(r"\bGompertz\b",                                                    _RE),
    "auteur_kannisto":           re.compile(r"\bKannisto\b",                                                    _RE),
    "auteur_coale_kisker":       re.compile(r"\bCoale[- ]Kisker\b",                                             _RE),
    "auteur_heligman_pollard":   re.compile(r"\bHeligman[- ]Pollard\b",                                         _RE),
    "auteur_brass":              re.compile(r"\bBrass\b",                                                       _RE),
    "auteur_eilers_marx":        re.compile(r"\bEilers[- ]Marx\b",                                              _RE),
    "auteur_whittaker_henderson": re.compile(r"\bWhittaker[- ]Henderson\b",                                     _RE),
    "auteur_denuit_goderniaux":  re.compile(r"\bDenuit[- ]Goderniaux\b",                                        _RE),
    "auteur_thérond":            re.compile(r"\bThérond\b",                                                     _RE),
}


# ─────────────────────────────────────────────────────────────────────────────
# Tags méthodologiques (clé doit commencer par "tag_" pour alimenter tags[])
# ─────────────────────────────────────────────────────────────────────────────
METHOD_TAGS: dict[str, re.Pattern] = {
    "tag_mortalite":      re.compile(r"\b(mortalité|mortalite|décès|deces)\b",                                  _RE),
    "tag_lissage":        re.compile(r"\b(lissage|smoothing|Whittaker|spline|p-spline)\b",                      _RE),
    "tag_prospectif":     re.compile(r"\b(prospecti(ve|f)|génération|generation|Lee[- ]Carter|cohorte)\b",      _RE),
    "tag_validation":     re.compile(r"\b(chi[- ]?2|chi-?carré|Kolmogorov|\bSMR\b|backtest|runs test)\b",       _RE),
    "tag_fermeture":      re.compile(r"\b(fermeture|grands? âges?|extrapolation|Kannisto|Denuit[- ]Goderniaux)\b", _RE),
    "tag_experience":     re.compile(r"\btables? d'expérience\b|\btable d'expérience\b",                        _RE),
    "tag_certification":  re.compile(r"\b(certification|certifié|certifie|agrément|Institut des Actuaires|\bIA\b)\b", _RE),
    "tag_solvabilite2":   re.compile(r"\b(Solvabilité ?2|Solvabilite ?2|\bSCR\b|best estimate|risk margin)\b",  _RE),
    "tag_formule_standard": re.compile(r"\bformule standard\b",                                                 _RE),
    "tag_ifrs17":         re.compile(r"\b(IFRS ?17|\bCSM\b|risk adjustment)\b",                                 _RE),
    "tag_longevite":      re.compile(r"\blongévité\b|\blongevite\b",                                            _RE),
    "tag_antiselection":  re.compile(r"\banti[- ]?sélection\b|\banti[- ]?selection\b",                          _RE),
    "tag_bcac":           re.compile(r"\bBCAC\b",                                                               _RE),
}


# ─────────────────────────────────────────────────────────────────────────────
# Régulatoire : articles, arrêtés, codes (un match → regulatory=True)
# ─────────────────────────────────────────────────────────────────────────────
REGULATORY: dict[str, re.Pattern] = {
    "art_a132_1":    re.compile(r"\bA132[- ]?1\b(?![0-9])",                                                     _RE),
    "art_a132_17":   re.compile(r"\bA132[- ]?17\b",                                                             _RE),
    "art_a132_18":   re.compile(r"\bA132[- ]?18\b",                                                             _RE),
    "art_a331_22":   re.compile(r"\bA331[- ]?22\b",                                                             _RE),
    "art_a335_1":    re.compile(r"\bA335[- ]?1\b",                                                              _RE),
    "art_r343_3":    re.compile(r"\bR343[- ]?3\b",                                                              _RE),
    "art_r351_2":    re.compile(r"\bR351[- ]?2\b",                                                              _RE),
    "arrete_date":   re.compile(r"\barrêté du \d{1,2}[/ -]\d{1,2}[/ -]\d{4}\b",                                 _RE),
    "loi_evin":      re.compile(r"\bloi Evin\b",                                                                _RE),
    "code_assur":    re.compile(r"\bCode des assurances\b",                                                     _RE),
    "arrete_minist": re.compile(r"\barrêté ministériel\b",                                                      _RE),
}


# ─────────────────────────────────────────────────────────────────────────────
# Tables réglementaires nommées (clé = libellé normalisé, alimente tables_referenced)
# ─────────────────────────────────────────────────────────────────────────────
TABLES: dict[str, re.Pattern] = {
    "TH_00_02":    re.compile(r"\bTH ?00[- ]?02\b",                                                             _RE),
    "TF_00_02":    re.compile(r"\bTF ?00[- ]?02\b",                                                             _RE),
    "TGH_05":      re.compile(r"\bTGH ?05\b",                                                                   _RE),
    "TGF_05":      re.compile(r"\bTGF ?05\b",                                                                   _RE),
    "TPRV_93":     re.compile(r"\bTPRV ?93\b",                                                                  _RE),
    "TD_TV_88_90": re.compile(r"\bTD ?[/-]? ?TV ?88[- ]?90\b|\bTD ?88[- ]?90\b|\bTV ?88[- ]?90\b",               _RE),
    "TD_TV_73_77": re.compile(r"\bTD ?[/-]? ?TV ?73[- ]?77\b|\bTD ?73[- ]?77\b|\bTV ?73[- ]?77\b",               _RE),
    "PM_PF_60_64": re.compile(r"\bPM ?[/-]? ?PF ?60[- ]?64\b",                                                  _RE),
    "BCAC_2010":   re.compile(r"\bBCAC ?2010\b",                                                                _RE),
    "BCAC_2013":   re.compile(r"\bBCAC ?2013\b",                                                                _RE),
}


# ─────────────────────────────────────────────────────────────────────────────
# Formules : symboles actuariels + LaTeX
# ─────────────────────────────────────────────────────────────────────────────
FORMULA_PATTERNS: list[re.Pattern] = [
    re.compile(r"\$[^$\n]+\$"),                       # LaTeX inline $...$
    re.compile(r"\\\[.*?\\\]", re.DOTALL),            # LaTeX display \[...\]
    re.compile(r"[µμ]_?\{?[a-zA-Z]+\}?"),             # force de mortalité
    re.compile(r"\b[qpLTSDdE]_\{?[a-zA-Z]+(\^[a-zA-Z]+)?\}?"),  # q_x, p_x, L_x, T_x, S_x, E_x^c, ...
]


def find_regex_matches(text: str) -> dict[str, list[str]]:
    """Applique AUTHORS + METHOD_TAGS + REGULATORY, retourne {clé: [matches]}.

    Reproduit le format du champ metadata.regex_matches dans chunks_enriched.json.
    Les patterns TABLES sont traités séparément (voir extract_tables).
    """
    out: dict[str, list[str]] = {}
    for registry in (AUTHORS, METHOD_TAGS, REGULATORY):
        for key, pat in registry.items():
            matches = pat.findall(text)
            if matches:
                # findall peut retourner des tuples si groupes capturants → on aplatit
                flat = [m if isinstance(m, str) else next((g for g in m if g), "") for m in matches]
                flat = [m for m in flat if m]
                if flat:
                    out[key] = flat
    return out


def extract_tags(regex_matches: dict[str, list[str]]) -> list[str]:
    """Dérive la liste des tags depuis regex_matches : clés `tag_*` → tag sans préfixe."""
    return sorted({key[4:] for key in regex_matches if key.startswith("tag_")})


def is_regulatory(regex_matches: dict[str, list[str]]) -> bool:
    """True ssi au moins un match dans le registre REGULATORY."""
    reg_keys = set(REGULATORY.keys())
    return any(key in reg_keys for key in regex_matches)


def extract_tables(text: str) -> list[str]:
    """Retourne la liste des tables réglementaires citées (clés du registre TABLES)."""
    return sorted([name for name, pat in TABLES.items() if pat.search(text)])


def count_formulas(text: str) -> int:
    """Compte les formules (LaTeX + symboles actuariels), dédup par span."""
    spans: set[tuple[int, int]] = set()
    for pat in FORMULA_PATTERNS:
        for m in pat.finditer(text):
            spans.add(m.span())
    # Déduplication grossière : si deux spans se chevauchent à >80%, on garde le plus long
    sorted_spans = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
    kept: list[tuple[int, int]] = []
    for s, e in sorted_spans:
        if not any(s >= ks and e <= ke for ks, ke in kept):
            kept.append((s, e))
    return len(kept)
