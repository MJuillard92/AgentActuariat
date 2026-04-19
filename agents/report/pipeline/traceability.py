"""
agents/report/pipeline/traceability.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Validator de traçabilité narratif → données.

Extrait tous les chiffres cités dans un texte rédigé par le LLM et vérifie
que chacun est présent (à ±tolérance) dans les données structurées
passées en entrée. Objectif : détecter les hallucinations et les
placeholders non résolus ([donnée non disponible], [key]).

Interface publique :
    extract_numbers(text)      -> list[float]
    collect_numbers(data)      -> set[float]
    validate_section(text, data, ...) -> TraceabilityResult
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class TraceabilityResult:
    ok:            bool
    numbers_cited: list[float]     = field(default_factory=list)
    untraceable:   list[float]     = field(default_factory=list)
    bad_tokens:    list[str]       = field(default_factory=list)   # [donnée non disponible], [key], etc.

    def feedback_for_retry(self) -> str:
        """Message chirurgical à renvoyer au LLM pour un retry ciblé."""
        parts = []
        if self.untraceable:
            parts.append(
                "Les valeurs suivantes ne sont pas traçables dans les données "
                "fournies (retire-les ou reformule SANS elles) : "
                + ", ".join(f"{v:g}" for v in self.untraceable)
            )
        if self.bad_tokens:
            parts.append(
                "Retire les tokens suivants du texte — ils trahissent un "
                "placeholder non résolu ou une donnée absente : "
                + ", ".join(self.bad_tokens)
            )
        return "\n".join(parts)


# ── Extraction de nombres ─────────────────────────────────────────────────────

# Matche : 71   |   3,14   |   95 %   |   0.0042   |   2 041 523
#   (groupes \d espace-séparés, virgule ou point décimal, % optionnel)
_NUMBER_RE = re.compile(
    r"""
    (?<![A-Za-z_])           # pas précédé d'une lettre (évite q_x, D_x...)
    (-?                      # signe optionnel
      \d{1,3}(?:[ \u00a0\u202f]\d{3})+   # groupe milliers : 2 041 523
      |\d+                   # ou entier simple
    )
    (?:[.,]\d+)?             # partie décimale optionnelle
    (?:\s?%)?                # signe pourcent optionnel
    """,
    re.VERBOSE,
)

_THOUSANDS_SEP = {" ", "\u00a0", "\u202f"}


def _parse_number(token: str) -> float | None:
    """Convertit '2 041 523,45' ou '95 %' en float."""
    s = token.strip().rstrip("%").strip()
    s = "".join(c for c in s if c not in _THOUSANDS_SEP)
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_numbers(text: str) -> list[float]:
    """Extrait tous les nombres cités dans un texte narratif."""
    out: list[float] = []
    for m in _NUMBER_RE.finditer(text or ""):
        val = _parse_number(m.group(0))
        if val is not None:
            out.append(val)
    return out


# ── Collecte récursive des nombres dans les données structurées ──────────────

def collect_numbers(data: Any) -> set[float]:
    """
    Parcourt récursivement un dict/list/valeur et collecte tous les nombres
    rencontrés. Utilisé comme référentiel pour la traçabilité.
    """
    seen: set[float] = set()
    _collect_into(data, seen)
    return seen


def _collect_into(obj: Any, seen: set[float]) -> None:
    if obj is None:
        return
    if isinstance(obj, bool):
        return                                  # pas un nombre cité en prose
    if isinstance(obj, (int, float)):
        seen.add(float(obj))
        return
    if isinstance(obj, str):
        for m in _NUMBER_RE.finditer(obj):
            v = _parse_number(m.group(0))
            if v is not None:
                seen.add(v)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_into(v, seen)
        return
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            _collect_into(v, seen)
        return


# ── Vérification de traçabilité avec tolérance ────────────────────────────────

def _is_close(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def _is_traceable(value: float, refs: Iterable[float], rel_tol: float, abs_tol: float) -> bool:
    """
    Une valeur est traçable si une valeur de référence :
    - correspond à ±tol
    - correspond à ±tol en pourcent (value × 100)    — ex. 0,748 ↔ 74,8 %
    - correspond à ±tol en fraction (value / 100)    — ex. 25 % ↔ 0,25
    - correspond à ±tol en ‰ (value / 1000)          — ex. 14,44 ‰ ↔ 0,01444

    Note : on ne tente PAS la conversion `value × 1000` pour éviter les
    faux positifs (un HR de 2,14 ne doit jamais matcher 2 140).
    """
    candidates = [value, value / 100.0, value * 100.0, value / 1000.0]
    for c in candidates:
        for r in refs:
            if _is_close(c, r, rel_tol, abs_tol):
                return True
    return False


# Tokens qui trahissent un bug de résolution — échec immédiat.
_BAD_TOKEN_RE = re.compile(
    r"\[donnée non disponible\]"
    r"|\[\s*[a-z_]+\s*\]"             # [key] brut
    r"|\{\{\s*[a-z_]+\s*\}\}",        # {{ key }} non substitué
    re.IGNORECASE,
)


# Whitelist EXACTE (pas de tolérance) — seulement des constantes purement
# syntaxiques qu'un actuaire écrit telles quelles sans les tirer de data_store.
_WHITELIST_EXACT: set[float] = {
    95.0, 99.0, 100.0,                     # pourcentages d'IC usuels
    0.01, 0.05, 0.1, 0.95, 0.99,            # seuils stats
    1.96, 2.576,                            # quantiles normaux usuels
    1000.0,                                 # base du permille
}


def _is_whitelisted(value: float) -> bool:
    # Comparaison exacte : 3,99 ne match PAS 4,0 ; 95 match 95
    return any(abs(value - w) < 1e-9 for w in _WHITELIST_EXACT)


def validate_section(
    text:    str,
    data:    dict,
    *,
    rel_tol: float = 0.02,                # ±2 % relatif
    abs_tol: float = 1e-4,                 # plancher pour arrondi d'affichage
    extra_refs: set[float] | None = None,
) -> TraceabilityResult:
    """
    Valide qu'un texte narratif ne cite que des chiffres présents dans `data`.

    Args:
        text        : texte rédigé par le LLM
        data        : dict de référence (ex. le bloc JSON injecté dans le prompt)
        rel_tol     : tolérance relative
        abs_tol     : tolérance absolue (pour les petits nombres)
        extra_refs  : nombres additionnels à considérer comme traçables
                      (ex. année en cours, âges de cohorte)

    Returns:
        TraceabilityResult
    """
    bad_tokens = sorted(set(_BAD_TOKEN_RE.findall(text or "")))
    cited  = extract_numbers(text or "")
    refs   = collect_numbers(data) | (extra_refs or set())

    untraceable = [
        v for v in cited
        if not _is_whitelisted(v) and not _is_traceable(v, refs, rel_tol, abs_tol)
    ]

    ok = not bad_tokens and not untraceable
    return TraceabilityResult(
        ok            = ok,
        numbers_cited = cited,
        untraceable   = untraceable,
        bad_tokens    = bad_tokens,
    )
