#!/usr/bin/env python3
"""
build_rag.py — Indexation du rapport de référence Winter & Associés dans ChromaDB.

Extrait le texte du PDF `Portefeuille/AF8796-TD3_v1.0.pdf` section par section,
sauvegarde chaque section sous forme de chunk markdown dans `knowledge_base/rag/chunks/`
pour inspection humaine, puis indexe dans la collection ChromaDB utilisée par
`tools/build_pdf/search_exemplars.py` (appelée à l'étape 03 du pipeline de rapport).

Cartographie PDF → section_id template YAML :
    Préambule                                         → preamble
    1. Les contrats + 2. Les données transmises       → data_submission
    3. La construction de la table                    → construction
    4.1 Décès observés et modélisés                   → obs_vs_modeled
    4.2 Comparaison avec la table antérieure          → precedent_comparison
    4.3 Positionnement réglementaire                  → regulatory_positioning
    5. Conclusion et recommandations                  → conclusion
    (+ 1 chunk synthétique style_guide)               → _style_guide

Usage :
    python knowledge_base/rag/build_rag.py
    python knowledge_base/rag/build_rag.py --dry-run         # pas d'indexation ChromaDB
    python knowledge_base/rag/build_rag.py --verify-only     # juste les queries test
    python knowledge_base/rag/build_rag.py --reset           # supprime puis réindexe

Idempotent : les IDs sont déterministes (source + section_id + chunk_index).
Une ré-exécution met à jour les chunks via `collection.upsert()`.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PDF_PATH     = _PROJECT_ROOT / "Portefeuille" / "AF8796-TD3_v1.0.pdf"
_CHUNKS_DIR   = _PROJECT_ROOT / "knowledge_base" / "rag" / "chunks"

# Aligné sur tools/build_pdf/search_exemplars.py — SURTOUT NE PAS DIVERGER.
_CHROMA_PATH  = _PROJECT_ROOT / "knowledge_base" / "rag" / "chroma_db"
_COLLECTION   = "exemplaires_actuariels"

# Bornes de taille d'un chunk de contenu.
_MIN_CHUNK_CHARS = 800
_MAX_CHUNK_CHARS = 2500


# ── Cartographie des sections ─────────────────────────────────────────────────
#
# Chaque entrée : (section_id, page_start, page_end, start_marker, end_marker)
# `start_marker` est le titre qui ouvre la section dans le flux de texte.
# `end_marker` est le titre qui ouvre la section suivante (exclu).
# `None` côté end_marker = dernière section couverte par la page_end.

_SECTION_MAP: list[dict] = [
    {
        "section_id":  "preamble",
        "page_start":  2,
        "page_end":    2,
        "start_marker": r"^PREAMBULE\b",
        "end_marker":   r"^SOMMAIRE\b",
    },
    {
        # Fusion section 1 (Les contrats) + 2 (Données transmises)
        "section_id":  "data_submission",
        "page_start":  4,
        "page_end":    7,
        "start_marker": r"^1\.\s*LES\s+CONTRATS\b",
        "end_marker":   r"^3\.\s*LA\s+CONSTRUCTION\s+DE\s+LA\s+TABLE\b",
    },
    {
        "section_id":  "construction",
        "page_start":  7,
        "page_end":    8,
        "start_marker": r"^3\.\s*LA\s+CONSTRUCTION\s+DE\s+LA\s+TABLE\b",
        "end_marker":   r"^4\.\s*COMMENTAIRES\b",
    },
    {
        "section_id":  "obs_vs_modeled",
        "page_start":  8,
        "page_end":    10,
        # 4.1 est le 1er commentaire — on prend tout "4. COMMENTAIRES" jusqu'à 4.2
        "start_marker": r"^4\.\s*COMMENTAIRES\b",
        "end_marker":   r"^4\.2\.\s*COMPARAISON\s+AVEC\s+LA\s+TABLE",
    },
    {
        "section_id":  "precedent_comparison",
        "page_start":  10,
        "page_end":    11,
        "start_marker": r"^4\.2\.\s*COMPARAISON\s+AVEC\s+LA\s+TABLE",
        "end_marker":   r"^4\.3\.\s*POSITIONNEMENT\s+PAR\s+RAPPORT",
    },
    {
        "section_id":  "regulatory_positioning",
        "page_start":  11,
        "page_end":    12,
        "start_marker": r"^4\.3\.\s*POSITIONNEMENT\s+PAR\s+RAPPORT",
        "end_marker":   r"^5\.\s*CONCLUSION\s+ET\s+RECOMMANDATIONS\b",
    },
    {
        "section_id":  "conclusion",
        "page_start":  13,
        "page_end":    13,
        "start_marker": r"^5\.\s*CONCLUSION\s+ET\s+RECOMMANDATIONS\b",
        "end_marker":   None,
    },
]


# ── Guide de style (extrait manuellement du rapport Winter) ──────────────────
#
# Ce chunk est ajouté en plus des 7 chunks de contenu. Il fournit au LLM les
# tournures, conventions typographiques, et le ton du rapport de référence
# SANS être noyé dans le contenu factuel.

_STYLE_GUIDE: str = """\
# Guide de style — rapport de certification de table de mortalité

Ce guide résume les tournures, conventions typographiques et la structure
narrative observées dans le rapport Winter & Associés AF8796-TD3 (2012),
référence pour la rédaction des rapports de certification de l'agent.

## Ton général

- Formel, descriptif, interprétatif en fin de section.
- Actuariel senior : chaque chiffre est commenté, pas seulement cité.
- Voix passive ou première personne du pluriel sobre (« on notera »,
  « on trouve »), jamais la première personne du singulier.
- Le narrateur évalue mais ne porte pas de jugement catégorique :
  « la méthodologie n'appelle pas de commentaire particulier »,
  « ce caractère linéaire tranche avec les analyses observées ».

## Tournures récurrentes à réutiliser

- « On notera [une tendance / une stabilité / un écart] » — pour introduire un
  constat factuel.
- « Il est précisé que [clarification technique] » — pour une mise au point.
- « Au global, [synthèse chiffrée] » — pour agréger après une analyse fine.
- « Le fait que [observation] met en évidence [conclusion] » — pour un lien
  cause-conséquence rigoureux.
- « La méthodologie n'appelle pas de commentaire particulier » — formule
  standard quand le processus suivi est conforme.
- « On peut en retenir [synthèse] » — pour conclure un paragraphe analytique.
- « Cette analyse est complétée par [contrôle additionnel] » — pour chaîner
  les contrôles.
- « Ce point est conforté par [second indicateur] » — pour corroborer.
- « Classiquement observé en pratique » — pour marquer l'alignement avec les
  références actuarielles.
- « Il est à noter que » — pour une mise en garde ou une nuance.

## Conventions typographiques

- **Décimales** : virgule française (0,289% et non 0.289%).
- **Milliers** : espace insécable fine (253 067 lignes, 780 411 années).
- **Pourcentages** : signe % collé au chiffre, précédé du nombre avec virgule
  décimale française si nécessaire (14,5%).
- **Intervalles de confiance** : `IC 95%` ou `intervalle de confiance à 95 %`.
- **Guillemets** : guillemets français « … » pour les noms de produits et
  citations.
- **Formules mathématiques** : notation LaTeX en ligne ($q_x = D_x / E_x$) ou
  en bloc centrée (\\[ … \\]) quand la formule est longue.
- **Références de tableaux et figures** : numérotées et citées en italique
  dans le texte : « cf. Tableau 7 », « cf. Figure 8 ».

## Référencement des tableaux et figures

- Chaque tableau et figure est introduit dans la prose AVANT son affichage :
  « La comparaison par tranches d'âges représentant près de 10 % de
  l'exposition au risque vient confirmer cette observation : »
- Suivi immédiatement du titre numéroté (« Tableau 7 — Comparaison des décès
  observés et des décès modélisés (par classe d'âges) »).
- Après le tableau, un paragraphe COMMENTE ce que l'on y voit : quels âges
  sont aberrants, quelle est l'ampleur de l'écart, etc. Jamais un tableau
  laissé sans interprétation.

## Structure d'une section type

1. **Phrase d'ouverture** qui pose l'objectif de la section et rappelle le
   contexte (« Afin de mesurer la prudence de la table d'expérience, l'approche
   retenue a consisté à … »).
2. **Exposé méthodologique** court (1-2 paragraphes) précisant la procédure
   de calcul ou le choix retenu.
3. **Résultats chiffrés** cités textuellement dans la prose, accompagnés
   d'une lecture actuarielle (prudence, cohérence, écart significatif).
4. **Tableau ou figure** avec caption numérotée.
5. **Commentaire du tableau** : quels âges, quelle amplitude, quelle
   interprétation pour la certification.
6. **Phrase de transition** vers la section suivante ou de synthèse locale.

## Transitions typiques entre sections

- « Sur cette base on effectue quelques statistiques descriptives présentées
  ci-après. »
- « Cette analyse est complétée, pour chaque année de la période considérée,
  par … »
- « L'analyse par âge conforte ce constat : »
- « Au global, [synthèse de ce qui précède]. »

## Conclusion type

- Rappel synthétique du périmètre et de la méthodologie.
- Reprise des **indicateurs clés chiffrés** (décès observés/modélisés, SMR,
  intervalle de confiance, abattements).
- Énoncé du **verdict** : table certifiable ou non, durée de validité
  (5 ans classiquement).
- **Domaine de validité** : produits concernés, évolutions acceptées.
- **Dispositif de suivi** : indicateurs à produire annuellement
  (obs/modélisé par classe d'âge, positionnement IC 95 %, sex-ratio).

## À éviter dans la rédaction

- Pas de « [donnée non disponible] » dans la prose : si une statistique manque,
  omettre la phrase entière plutôt que de la signaler.
- Pas de placeholder brut type `{{ … }}` ou `[…]` : toujours résoudre ou
  supprimer la phrase.
- Pas de chiffre en notation scientifique (`2.14e-3`) dans le corps de texte :
  écrire « 0,00214 » ou « 2,14 ‰ ».
- Pas de superlatifs sans justification chiffrée (« très bonne prudence »
  doit être soutenu par un ratio, un IC, etc.).
- Pas de paraphrase plate du tableau : commenter, ne pas recopier.
"""


# ── Extraction PDF ────────────────────────────────────────────────────────────

def extract_pages(pdf_path: Path) -> list[str]:
    """Retourne une liste de textes, indexée par numéro de page - 1."""
    try:
        import pdfplumber
    except ImportError:
        sys.exit(
            "Erreur : pdfplumber non installé.\n"
            "    pip install pdfplumber"
        )

    pages: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            pages.append((page.extract_text() or "").strip())
    return pages


# ── Nettoyage ─────────────────────────────────────────────────────────────────

# Footer Winter : « 15/05/2012 – AF8796-TD3 CONFIDENTIEL WINTER & Associés - Page X/14 »
_FOOTER_RE  = re.compile(r"^\s*\d{1,2}/\d{2}/\d{4}.*CONFIDENTIEL.*WINTER.*Page\s*\d+/\d+\s*$", re.IGNORECASE)
_PAGE_RE    = re.compile(r"^\s*Page\s*\d+\s*(/\d+)?\s*$", re.IGNORECASE)
_MULTISPACE = re.compile(r" {2,}")
_MULTINL    = re.compile(r"\n{3,}")

def clean_page(text: str) -> str:
    """Retire footer Winter, numéros de page isolés, espaces redondants."""
    kept = []
    for line in text.split("\n"):
        if _FOOTER_RE.match(line) or _PAGE_RE.match(line):
            continue
        kept.append(line.rstrip())
    out = "\n".join(kept)
    out = _MULTISPACE.sub(" ", out)
    out = _MULTINL.sub("\n\n", out)
    return out.strip()


# ── Extraction par section ────────────────────────────────────────────────────

def extract_section_text(pages: list[str], spec: dict) -> str:
    """
    Concatène les pages [page_start, page_end] puis découpe entre `start_marker`
    (inclus) et `end_marker` (exclu).
    """
    p_start = spec["page_start"] - 1  # 0-based
    p_end   = spec["page_end"]        # exclusive slice end
    blob    = "\n\n".join(clean_page(pages[i]) for i in range(p_start, p_end))

    # Découpe au start_marker
    lines = blob.split("\n")
    start_idx = None
    for i, ln in enumerate(lines):
        if re.match(spec["start_marker"], ln.strip(), re.IGNORECASE):
            start_idx = i
            break
    if start_idx is None:
        raise RuntimeError(
            f"Section {spec['section_id']} : start_marker introuvable "
            f"({spec['start_marker']})"
        )

    # Découpe au end_marker
    end_idx = len(lines)
    if spec["end_marker"]:
        for j in range(start_idx + 1, len(lines)):
            if re.match(spec["end_marker"], lines[j].strip(), re.IGNORECASE):
                end_idx = j
                break

    section_text = "\n".join(lines[start_idx:end_idx]).strip()
    return section_text


# ── Découpage en sous-chunks ──────────────────────────────────────────────────

def split_into_chunks(
    text: str,
    min_size: int = _MIN_CHUNK_CHARS,
    max_size: int = _MAX_CHUNK_CHARS,
) -> list[str]:
    """
    Découpe un texte en chunks respectant les paragraphes.
    Objectif : chaque chunk ∈ [min_size, max_size] chars.
    Un paragraphe n'est jamais coupé au milieu.
    """
    if len(text) <= max_size:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # Si le paragraphe seul dépasse max_size, on le prend tel quel
        # plutôt que de casser la sémantique — mieux vaut un chunk un peu long.
        if len(para) > max_size and not current:
            chunks.append(para)
            continue

        # Essai d'ajout au chunk courant
        candidate = (current + "\n\n" + para).strip() if current else para

        if len(candidate) <= max_size:
            current = candidate
        else:
            # Le chunk courant est à flush s'il est assez gros
            if len(current) >= min_size:
                chunks.append(current)
                current = para
            else:
                # Trop petit, on continue d'agréger quitte à dépasser légèrement
                current = candidate

    if current:
        chunks.append(current)

    return chunks


# ── Sauvegarde markdown pour inspection ───────────────────────────────────────

def save_chunks_md(section_id: str, chunks: list[str], spec: dict, out_dir: Path) -> None:
    """Écrit les chunks d'une section dans un fichier markdown unique."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{section_id}.md"

    lines = [
        f"# {section_id}",
        "",
        f"Source : `{_PDF_PATH.name}` — pages {spec.get('page_start', '?')}–{spec.get('page_end', '?')}",
        "",
    ]
    for i, chunk in enumerate(chunks):
        lines += [
            f"## Chunk {i} ({len(chunk)} chars)",
            "",
            chunk,
            "",
        ]
    out.write_text("\n".join(lines), encoding="utf-8")


def save_style_guide_md(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_style_guide.md").write_text(_STYLE_GUIDE, encoding="utf-8")


# ── Indexation ChromaDB ───────────────────────────────────────────────────────

def _chunk_id(section_id: str, chunk_index: int) -> str:
    """ID déterministe pour permettre l'upsert idempotent."""
    return f"{_PDF_PATH.stem}__{section_id}__{chunk_index:02d}"


def _ensure_collection(reset: bool = False):
    try:
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    except ImportError:
        sys.exit("Erreur : chromadb non installé.  pip install chromadb")

    _CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    ef     = DefaultEmbeddingFunction()

    if reset:
        try:
            client.delete_collection(_COLLECTION)
        except Exception:
            pass

    try:
        coll = client.get_collection(name=_COLLECTION, embedding_function=ef)
    except Exception:
        coll = client.get_or_create_collection(name=_COLLECTION, embedding_function=ef)

    return coll


def index_chunks(section_id: str, chunks: list[str], spec: dict, collection) -> int:
    """Upsert les chunks d'une section. Retourne le nombre upserté."""
    ids, docs, metas = [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(_chunk_id(section_id, i))
        docs.append(chunk)
        metas.append({
            # Clés demandées par INSTRUCTION_RAG.md
            "section_id":  section_id,
            "source":      _PDF_PATH.name,
            "chunk_type":  "content",
            "page_start":  spec["page_start"],
            "page_end":    spec["page_end"],
            "chunk_index": i,
            "report_type": "mortality_certification",
            # Compat search_exemplars.py (qui lit `section`, `rapport_id`, `type_rapport`)
            "section":     section_id,
            "rapport_id":  _PDF_PATH.stem,
            "type_rapport": "certification",
            "produit":     "temporaire_deces",
            "methode_lissage": "makeham+whittaker",
            "qualite_exemplaire": "high",
        })
    collection.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def index_style_guide(collection) -> int:
    cid = _chunk_id("_style_guide", 0)
    collection.upsert(
        ids=[cid],
        documents=[_STYLE_GUIDE],
        metadatas=[{
            "section_id":  "_style_guide",
            "source":      _PDF_PATH.name,
            "chunk_type":  "style_guide",
            "page_start":  0,
            "page_end":    0,
            "chunk_index": 0,
            "report_type": "mortality_certification",
            # Compat search_exemplars.py
            "section":     "_style_guide",
            "rapport_id":  _PDF_PATH.stem,
            "type_rapport": "certification",
            "produit":     "temporaire_deces",
            "qualite_exemplaire": "high",
        }],
    )
    return 1


# ── Vérification ──────────────────────────────────────────────────────────────

_TEST_QUERIES: dict[str, str] = {
    "preamble": "introduction d'un rapport de certification de table de mortalité",
    "data_submission": "statistiques descriptives des données, exposition et décès par âge",
    "construction": "choix de la méthode de lissage Makeham raccordement table réglementaire",
    "obs_vs_modeled": "comparaison décès observés et décès modélisés avec intervalles de confiance",
    "precedent_comparison": "comparaison entre deux tables d'expérience successives évolution de la prudence",
    "regulatory_positioning": "positionnement par rapport aux tables TH TF abattements et régression logit",
    "conclusion": "conclusion et recommandations d'un rapport de certification, domaine de validité",
    "_style_guide": "tournures récurrentes d'un rapport actuariel, conventions typographiques",
}


def verify_collection(collection) -> dict:
    """
    Pour chaque section_id, lance une query test et retourne :
    {section_id: {top_id, top_section, distance, length}}
    """
    report: dict = {}
    for section_id, query in _TEST_QUERIES.items():
        res = collection.query(
            query_texts=[query],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        ids   = res.get("ids",       [[]])[0]
        docs  = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        if ids:
            report[section_id] = {
                "top_id":      ids[0],
                "top_section": (metas[0] or {}).get("section_id", "?"),
                "distance":    round(dists[0], 4),
                "length":      len(docs[0] or ""),
                "ok":          (metas[0] or {}).get("section_id") == section_id,
            }
        else:
            report[section_id] = {"top_id": None, "ok": False}
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Indexation RAG du rapport Winter dans ChromaDB.")
    parser.add_argument("--dry-run",     action="store_true", help="Génère les chunks .md sans indexer")
    parser.add_argument("--verify-only", action="store_true", help="Saute l'indexation, lance juste les queries test")
    parser.add_argument("--reset",       action="store_true", help="Supprime la collection puis réindexe")
    args = parser.parse_args()

    if not _PDF_PATH.exists():
        sys.exit(f"Erreur : PDF introuvable : {_PDF_PATH}")

    if args.verify_only:
        coll = _ensure_collection(reset=False)
        report = verify_collection(coll)
        print_report(coll.count(), {}, report)
        return

    # 1. Extraire toutes les pages
    print(f"[build_rag] Extraction PDF : {_PDF_PATH.name}")
    pages = extract_pages(_PDF_PATH)
    print(f"[build_rag]   {len(pages)} pages extraites")

    # 2. Découper section par section, sauver les markdown
    section_chunks: dict[str, tuple[dict, list[str]]] = {}
    for spec in _SECTION_MAP:
        sid = spec["section_id"]
        raw = extract_section_text(pages, spec)
        chunks = split_into_chunks(raw)
        section_chunks[sid] = (spec, chunks)
        save_chunks_md(sid, chunks, spec, _CHUNKS_DIR)
        lengths = [len(c) for c in chunks]
        print(
            f"[build_rag]   {sid:<24s}  "
            f"{len(chunks)} chunk(s)  "
            f"lens={lengths}  "
            f"p{spec['page_start']}-{spec['page_end']}"
        )

    # Style guide
    save_style_guide_md(_CHUNKS_DIR)
    print(f"[build_rag]   _style_guide             1 chunk       len=[{len(_STYLE_GUIDE)}]  (manuel)")

    if args.dry_run:
        print("[build_rag] --dry-run : pas d'indexation ChromaDB")
        print(f"[build_rag] Chunks MD écrits dans : {_CHUNKS_DIR}")
        return

    # 3. Indexer dans ChromaDB
    coll = _ensure_collection(reset=args.reset)
    counts_by_section: dict[str, int] = {}

    for sid, (spec, chunks) in section_chunks.items():
        n = index_chunks(sid, chunks, spec, coll)
        counts_by_section[sid] = n
    counts_by_section["_style_guide"] = index_style_guide(coll)

    # 4. Vérification
    report = verify_collection(coll)
    print_report(coll.count(), counts_by_section, report)


def print_report(total_count: int, counts_by_section: dict, verify: dict) -> None:
    print()
    print("━" * 68)
    print(f"Collection ChromaDB '{_COLLECTION}' — {total_count} chunks au total")
    print(f"Chemin : {_CHROMA_PATH}")
    print("━" * 68)

    if counts_by_section:
        print("\nChunks indexés (upsert) :")
        for sid, n in counts_by_section.items():
            print(f"  {sid:<24s}  {n} chunk(s)")

    print("\nQueries de vérification :")
    for sid, info in verify.items():
        mark = "✓" if info.get("ok") else "✗"
        if info.get("top_id"):
            print(
                f"  {mark} {sid:<24s}  "
                f"→ {info['top_section']:<24s}  "
                f"dist={info['distance']}  len={info['length']}"
            )
        else:
            print(f"  ✗ {sid:<24s}  → aucun résultat")

    n_ok = sum(1 for v in verify.values() if v.get("ok"))
    n_total = len(verify)
    print(f"\nRésumé : {n_ok}/{n_total} sections retournent le bon chunk en top-1.")


if __name__ == "__main__":
    main()
