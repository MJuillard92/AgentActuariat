"""
scripts/redact_rag_chunks.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Purge les chiffres et noms propres clients des chunks RAG
(knowledge_base/rag/chunks/*.md) pour éviter que le LLM rédacteur les copie
dans le rapport généré.

Stratégie : protect-then-purge
  1. Marquer les patterns à PROTÉGER (réglementaire, méthodologique) avec
     des sentinelles temporaires.
  2. Appliquer les regex de purge sur le texte restant.
  3. Restaurer les sentinelles.

Usage :
    python scripts/redact_rag_chunks.py --dry-run     # affiche le diff, n'écrit rien
    python scripts/redact_rag_chunks.py               # écrit en place
    python scripts/redact_rag_chunks.py --file preamble.md  # un seul chunk
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHUNKS_DIR = _PROJECT_ROOT / "knowledge_base" / "rag" / "chunks"


# ── Patterns à PROTÉGER (gardés intacts dans la purge) ────────────────────
#
# Ordre important : on protège d'abord les patterns longs (qui pourraient
# contenir des nombres protégés par patterns plus courts).
#
PROTECTED_PATTERNS: list[str] = [
    # Références réglementaires
    r"Art\.\s*A\.\s*335-1\s+du\s+Code\s+des\s+Assurances",
    r"Art\.\s*A\.\s*335-1",
    r"Code\s+des\s+Assurances",

    # Tables réglementaires (variantes orthographiques)
    r"\bTH\s*00\s*/\s*02\b",
    r"\bTF\s*00\s*/\s*02\b",
    r"\bTHF\s*00\s*/\s*02\b",
    r"\bTH[-/]TF\s*00[-/]02\b",
    r"\bTPRV\s*93\b",
    r"\bTH00/02\b",
    r"\bTF\s*00/02\b",
    r"\bTHF\s*00/02\b",

    # Constantes méthodologiques (IC, plages standards)
    r"\bIC\s*(?:à\s*)?95\s*%",
    r"intervalles?\s+de\s+confiance\s+(?:à\s+)?95\s*%",
    r"\b95\s*%(?:\s+(?:IC|de\s+confiance))",   # "95% IC", "95% de confiance"
    r"\b110\s*ans\b",                            # seuil aberrant
    r"\b45[-\s]*64\s*ans\b",                     # plage Makeham
    r"\b2\s*ans\s+d['']ancienneté",              # sélection médicale
    r"\b5\s*ans\b(?=\s+(?:de\s+validité|d['']utilisation))",

    # Méthodes statistiques / actuarielles (noms propres techniques)
    r"\bWhittaker[-\s]Henderson\b",
    r"\bWhittaker\b",
    r"\bMakeham\b",
    r"\bKaplan[-\s]Meier\b",
    r"\bGompertz\b",
    r"\bSchönfeld\b",
    r"\bmodèle\s+de\s+Cox\b",
    r"\bcalcul\s+mensuel\b",
    r"\btest\s+du\s+chi-?(?:deux|²)\b",

    # Notations mathématiques (préserver q_x, D_x, E_x, SMR…)
    r"\bq_x\b",
    r"\bD_x\b",
    r"\bE_x\b",
    r"\bSMR\b",
    r"\bq[xX]\b",
]


# ── Patterns de PURGE (appliqués dans l'ordre) ────────────────────────────
#
REDACTIONS: list[tuple[str, str]] = [
    # 1. Noms propres clients / certificateurs (en premier — avant
    #    d'enlever les nombres qui pourraient en faire partie : "AF8796-TD3")
    (r"\bAllianz\b",                              "[CLIENT]"),
    (r"WINTER\s*&\s*Associés",                    "[CERTIFICATEUR]"),
    (r"\bWINTER\b",                               "[CERTIFICATEUR]"),

    # 2. Codes de contrats (liste détaillée du préambule du portefeuille Allianz)
    (r"\bAGF[A-Z][A-Za-z]*\b",                    "[PRODUIT]"),
    (r"\bChorus[A-Z][A-Za-z]*\b",                 "[PRODUIT]"),
    (r"\b(?:AssDCAGF|HERMES|NM_PFA_TP|Stabila|TVA|Variato5?)\b", "[PRODUIT]"),

    # 3. Références internes d'études (AF8796-TD1, etc.)
    (r"\bAF\d{4}-TD\d+\b",                        "[REF_ETUDE]"),
    (r"\bn°AF\d{4}-TD\d+\b",                      "n°[REF_ETUDE]"),

    # 4. Noms de fichiers : « XXX.txt », « XXX.csv », etc.
    (r"«\s*[A-Za-z0-9_.\-]+\.(?:txt|csv|xlsx?|json)\s*»",  "« XXX »"),

    # 5. Identifiants techniques (champs de la base : DECES, SORTIE, CLINAISS, …)
    #    Risque : trop large, on laisse pour l'instant. Le LLM identifie en
    #    général les vrais champs depuis le data_store.

    # 6. Dates pleines : JJ/MM/YYYY
    (r"\b\d{1,2}/\d{1,2}/\d{4}\b",                "JJ/MM/YYYY"),

    # 7. Périodes calendaires : [YYYY-YYYY] ou (YYYY à YYYY) ou "de YYYY à YYYY"
    (r"\b(?:19|20)\d{2}[-–](?:19|20)\d{2}\b",     "YYYY-YYYY"),
    (r"\bde\s+(?:19|20)\d{2}\s+à\s+(?:19|20)\d{2}\b",  "de YYYY à YYYY"),
    (r"\ben\s+(?:19|20)\d{2}\b",                  "en YYYY"),

    # 8. Années isolées (19xx, 20xx) — on tape large, le LLM doit utiliser
    #    les vraies années du data_store.
    (r"\b(?:19|20)\d{2}\b",                       "YYYY"),

    # 9. Pourcentages avec valeur. Ne touche PAS "IC 95 %" (protégé).
    (r"-?\b\d{1,3}(?:[.,]\d+)?\s*%",              "XXX %"),

    # 10. Nombres groupés avec espace (français) : 1 546, 253 067, 1 546,33
    (r"\b\d{1,3}(?:\s\d{3})+(?:[.,]\d+)?\b",      "XXX"),

    # 11. Nombres ≥ 100 (sans groupement) — peut être un entier ou décimal.
    (r"\b\d{3,}(?:[.,]\d+)?\b",                   "XXX"),

    # 12. Probabilités / p-values / ratios décimaux < 1 (0,96 ; 0,289 ; 1,5)
    (r"\b\d+,\d+\b",                              "XXX"),

    # 13. Décès chiffrés "X décès" pour X petit (1-99 cas particuliers)
    #     On NE touche PAS car les âges (44 ans, 64 ans) seraient impactés.
    #     La règle "nombre ≥ 100" gère le gros volume.
]


# ── Logique principale ─────────────────────────────────────────────────────

def redact_text(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Applique la purge. Retourne (texte purgé, liste des changements).

    Algorithme :
      1. Remplace les patterns protégés par des sentinelles uniques.
      2. Applique les redactions sur le texte avec sentinelles.
      3. Restaure les patterns protégés à leur emplacement original.
    """
    # Étape 1 : protéger
    protect_map: dict[str, str] = {}
    out = text
    for pat in PROTECTED_PATTERNS:
        def _sub_protect(m, _pat=pat):
            sentinel = f"\x00PROT{len(protect_map):05d}\x00"
            protect_map[sentinel] = m.group(0)
            return sentinel
        out = re.sub(pat, _sub_protect, out, flags=re.IGNORECASE)

    # Étape 2 : purger
    changes: list[tuple[str, str]] = []
    for pat, repl in REDACTIONS:
        for m in re.finditer(pat, out):
            changes.append((m.group(0), repl))
        out = re.sub(pat, repl, out)

    # Étape 3 : restaurer
    for sentinel, original in protect_map.items():
        out = out.replace(sentinel, original)

    return out, changes


def show_diff(filename: str, before: str, after: str) -> None:
    """Affiche le diff coloré (style ANSI rouge/vert)."""
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        n=1,
    )
    for line in diff:
        if line.startswith("+++") or line.startswith("---"):
            sys.stdout.write(f"\033[1m{line}\033[0m")
        elif line.startswith("+"):
            sys.stdout.write(f"\033[32m{line}\033[0m")   # green
        elif line.startswith("-"):
            sys.stdout.write(f"\033[31m{line}\033[0m")   # red
        elif line.startswith("@@"):
            sys.stdout.write(f"\033[36m{line}\033[0m")   # cyan
        else:
            sys.stdout.write(line)


def process_file(path: Path, dry_run: bool, show_changes: bool) -> dict:
    """Traite un fichier, retourne stats."""
    before = path.read_text(encoding="utf-8")
    after, changes = redact_text(before)

    if show_changes and changes:
        print(f"\n━━━ {path.name} ━━━ ({len(changes)} substitutions)")
        # Échantillon des 8 premiers changements
        from collections import Counter
        c = Counter([(orig[:40], repl) for orig, repl in changes])
        for (orig, repl), n in c.most_common(8):
            print(f"  {n:3}× {orig!r:46} → {repl!r}")
        if len(c) > 8:
            print(f"  ... et {len(c) - 8} autres patterns")

    if dry_run:
        if before != after:
            show_diff(path.name, before, after)
    else:
        if before != after:
            path.write_text(after, encoding="utf-8")

    return {
        "file":       path.name,
        "changed":    before != after,
        "n_changes":  len(changes),
        "len_before": len(before),
        "len_after":  len(after),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Affiche le diff sans écrire les fichiers")
    ap.add_argument("--file", default=None,
                    help="Traite uniquement ce chunk (ex: preamble.md)")
    ap.add_argument("--no-show", action="store_true",
                    help="Ne pas lister les patterns substitués")
    args = ap.parse_args()

    if args.file:
        paths = [_CHUNKS_DIR / args.file]
        if not paths[0].exists():
            print(f"ERROR: {paths[0]} introuvable", file=sys.stderr)
            return 1
    else:
        paths = sorted(_CHUNKS_DIR.glob("*.md"))

    if not paths:
        print(f"Aucun .md dans {_CHUNKS_DIR}", file=sys.stderr)
        return 1

    stats = []
    for p in paths:
        s = process_file(p, dry_run=args.dry_run, show_changes=not args.no_show)
        stats.append(s)

    print(f"\n{'─' * 78}")
    n_changed = sum(1 for s in stats if s["changed"])
    n_subs = sum(s["n_changes"] for s in stats)
    print(f"Traité : {len(stats)} fichiers, {n_changed} modifiés, {n_subs} substitutions")
    if args.dry_run:
        print("(mode --dry-run : aucun fichier écrit)")
    else:
        print("Fichiers écrits en place. Pense à rebuilder le ChromaDB :")
        print("    python knowledge_base/rag/build_rag.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
