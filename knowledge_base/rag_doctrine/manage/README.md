# manage/ — Gestion du corpus doctrinal RAG

Outils CLI + UI pour ajouter, lister et supprimer des documents dans
`knowledge_base/rag_doctrine/chunks_enriched.json` (source de vérité) et
reconstruire l'index FAISS associé.

## Architecture

```
manage/
├── _patterns.py    # regex actuarielles (auteurs, tags, articles, tables)
├── _extract.py     # PDF (PyMuPDF) + DOCX (python-docx) → list[Block]
├── _chunking.py    # group_by_section + chunk_blocks
├── _enrich.py      # metadata complète + formules
├── _indexer.py     # IO chunks + rebuild FAISS atomique
├── cli.py          # CLI argparse + logique métier réutilisable
├── ui.py           # onglet Dash "Doctrine RAG"
└── README.md
```

## CLI

```bash
# Lister les documents indexés
python -m knowledge_base.rag_doctrine.manage.cli list
python -m knowledge_base.rag_doctrine.manage.cli list --doc-id D03 --format json

# Ajouter un dossier de PDF/DOCX (dry-run pour preview)
python -m knowledge_base.rag_doctrine.manage.cli add --dir /path/to/folder --dry-run
python -m knowledge_base.rag_doctrine.manage.cli add --dir /path/to/folder

# Forcer un doc_id (un seul fichier dans le dossier)
python -m knowledge_base.rag_doctrine.manage.cli add --dir /path/to/folder --doc-id D43

# Remplacer un doc existant (même fingerprint ou même doc_id)
python -m knowledge_base.rag_doctrine.manage.cli add --dir /path/to/folder --force

# Supprimer un document
python -m knowledge_base.rag_doctrine.manage.cli delete --doc-id D13
python -m knowledge_base.rag_doctrine.manage.cli delete --doc-id D13 --confirm  # sans prompt

# Reconstruire l'index FAISS depuis chunks_enriched.json (utile après git pull)
python -m knowledge_base.rag_doctrine.manage.cli rebuild
```

## UI

L'onglet **Doctrine RAG** est ajouté à `canvas_app.py`. Pour le lancer :

```bash
python canvas_app.py
```

Layout :
- **Header** : dropzone drag-drop multi-fichiers PDF/DOCX + bouton Refresh
- **Gauche** : liste des documents (doc_id + nb chunks + titre)
- **Droite** : tous les chunks du document sélectionné (texte complet + badges
  tags / regulatory / formules / word_count) + bouton Supprimer

L'upload via UI déclenche exactement la même chaîne que la CLI `add` (extract
→ chunk → enrich → embed → FAISS atomique).

## Pipeline d'ingestion (interne)

1. **SHA256** du fichier source → `metadata.source_fingerprint` (idempotence)
2. **Extraction** PDF/DOCX → `list[Block]` (text, page, heading_level, block_index)
3. **Chunking** par section (heading ≤ 2) avec respect des paragraphes
   - target 2000 chars, max 2500, min 400, merge du dernier si trop court
4. **Enrichissement** : regex auteurs/tags/articles/tables, comptage formules
5. **Build** chunks avec UUID, section_id incrémental, schéma compatible
6. **Embedding** via `paraphrase-multilingual-MiniLM-L12-v2` (dim 384, figé)
7. **Écriture atomique** : `faiss.bin.tmp` + `meta.json.tmp` puis `os.replace`

## Schéma chunk

Strictement compatible avec le schéma initial (`HybridRetriever` non modifié).
Trois champs ajoutés dans `metadata` :

| Champ | Type | Usage |
|---|---|---|
| `source_fingerprint` | str (SHA256 hex) | Idempotence — skip si déjà ingéré |
| `source_filename` | str | Affichage UI |
| `ingested_at` | str (ISO 8601 UTC) | Traçabilité |

## Idempotence

Re-ingérer le même fichier (SHA256 identique) est un **no-op** (skip + warning).
Pour forcer le ré-ingestion : `--force` (supprime d'abord l'ancien doc).

## Backups

Chaque modification de `chunks_enriched.json` crée un backup horodaté
(`chunks_enriched.json.bak.YYYYMMDD-HHMMSS`). Non purgés automatiquement.

## Concurrence et cache retriever

L'écriture atomique garantit qu'aucun process ne voit un état partiel
(`faiss.bin` toujours cohérent avec `meta.json`). **Cependant**, le retriever
de `tools/conversation/search_doctrine.py` cache l'index en RAM au premier
appel — il faut **redémarrer** le process consommateur (canvas_app inclus)
pour qu'il prenne en compte les nouveaux chunks lors d'une recherche.

L'UI affiche correctement les chunks juste après ingestion (lecture directe
de `chunks_enriched.json`), mais la **recherche RAG** depuis le tab `Rapport
guidé` continuera d'utiliser le cache jusqu'au prochain démarrage.

## Limitations connues

- `doc_id` limité à `D999` (raise NotImplementedError au-delà).
- `--doc-id` interdit si le dossier contient plusieurs fichiers.
- Pas de purge automatique des `.bak.*` — à faire manuellement.
- Détection heading PDF heuristique (font-size + majuscules) : un PDF sans
  structure typographique claire donnera une seule section géante (les chunks
  restent corrects, seuls les `section_title` deviennent génériques).
