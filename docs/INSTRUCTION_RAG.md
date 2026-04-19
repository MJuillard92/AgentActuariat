# Instruction — Construction du RAG à partir du rapport de référence

## Contexte

Le pipeline de rédaction (étape 03, `agents/report/pipeline/_03_completion_plan.py`)
interroge ChromaDB pour récupérer des exemples de rédaction par section.
**La base ChromaDB est actuellement vide**, ce qui signifie que le LLM rédige sans
aucune référence stylistique.

Le rapport de référence à indexer est :
`/Users/macbook14/Python_projects/AgentActuariat/Portefeuille/AF8796-TD3_v1.0.pdf`

Il s'agit d'un rapport Winter & Associés de certification de table de mortalité
d'expérience (14 pages, 2012, contrat temporaire décès Allianz). C'est exactement
le livrable que l'agent doit reproduire en style et en structure.

## Objectif

Construire un script d'indexation qui :
1. Extrait le texte du PDF section par section.
2. Produit deux types de chunks : **chunks de contenu** par section et un **guide de style** global.
3. Les indexe dans ChromaDB avec les bonnes métadonnées.
4. Est idempotent (peut être relancé sans dupliquer les chunks).

## Cartographie sections du rapport ↔ sections du template YAML

Le rapport de référence contient 7 sections qui doivent être mappées ainsi :

| Section du PDF de référence | `section_id` dans le template | Page(s) |
|---|---|---|
| Préambule | `preamble` | 2 |
| 1. Les contrats | `data_submission` | 4 |
| 2. Les données transmises (2.1 + 2.2) | `data_submission` | 4–7 |
| 3. La construction de la table | `construction` | 7–8 |
| 4.1 Décès observés et décès modélisés | `obs_vs_modeled` | 8–10 |
| 4.2 Comparaison avec la table d'expérience précédente | `precedent_comparison` | 10–11 |
| 4.3 Positionnement par rapport aux tables réglementaires | `regulatory_positioning` | 11–12 |
| 5. Conclusion et recommandations (5.1 + 5.2) | `conclusion` | 13 |

**À ignorer** pour l'indexation : page de garde, sommaire, annexe (table q_x brute).

## Stratégie de chunking

### 1. Chunks de contenu (indexation par section)

- **Un chunk par section** du tableau ci-dessus, texte intégral (pas de troncature à 600 chars).
- Les sections 2 (2.1 et 2.2) doivent être concaténées en un seul chunk `data_submission`.
- Même chose pour section 5 (5.1 et 5.2) concaténées en `conclusion`.
- **Longueur cible par chunk** : entre 800 et 2500 caractères. Si une section dépasse 2500
  caractères, la découper en sous-chunks en respectant les paragraphes (jamais au milieu d'une phrase).
- Les formules mathématiques mal extraites par le parseur PDF doivent être nettoyées ou marquées
  explicitement `[formule mathématique omise]`.
- Les références de tableaux et figures doivent être conservées (ex : « Tableau 7 – ... »).

### 2. Guide de style global

Créer un chunk spécial `style_guide` (metadata `section_id="_style_guide"`) qui contient :
- Les 15–20 tournures récurrentes repérées dans le rapport (« On notera que », « Il est précisé que »,
  « Au global », « Le caractère linéaire de... », « On peut en retenir », etc.)
- Les règles de présentation des chiffres (virgule décimale française, espace fine pour les milliers,
  pourcentages avec virgule : « 14,5% »).
- La convention de référencement des tableaux et figures.
- Le ton général : formel, descriptif, interprétatif à la fin de chaque section.
- Les transitions typiques entre sections.

Ce guide doit être extrait **manuellement** (ou semi-manuellement avec un LLM) à partir
du rapport, pas généré automatiquement.

## Métadonnées ChromaDB

Chaque chunk doit porter les metadata suivantes :

```python
{
    "section_id": "preamble",              # ou "data_submission", "construction", etc.
    "source": "AF8796-TD3_v1.0.pdf",
    "chunk_type": "content",               # ou "style_guide"
    "page_start": 2,
    "page_end": 2,
    "chunk_index": 0,                      # position dans la section si multi-chunks
    "report_type": "mortality_certification",
}
```

## Organisation des fichiers

```
knowledge_base/
  rag/
    build_rag.py              ← script d'indexation (à créer)
    chunks/                   ← chunks intermédiaires sauvegardés en .md pour inspection
      preamble.md
      data_submission.md
      construction.md
      obs_vs_modeled.md
      precedent_comparison.md
      regulatory_positioning.md
      conclusion.md
      _style_guide.md
    chroma_db/                ← base ChromaDB persistée (si pas déjà configurée ailleurs)
```

## Logique du script `build_rag.py`

1. **Vérifier** où ChromaDB est déjà configurée dans le projet (chercher les appels
   `chromadb.PersistentClient` et `collection.add`). Utiliser le même chemin et le même
   nom de collection que l'étape 03 du pipeline pour éviter un décalage.
2. **Extraire** le texte du PDF avec `pdfplumber` (meilleur que `pypdf` pour conserver
   la mise en page).
3. **Découper** en sections selon la cartographie ci-dessus (utiliser les titres comme
   points de coupe, par exemple « PREAMBULE », « 1. LES CONTRATS », etc.).
4. **Nettoyer** chaque chunk : retirer les en-têtes de page (« 15/05/2012 – AF8796-TD3
   CONFIDENTIEL WINTER & Associés - Page X/14 »), les notes de bas de page parasites,
   les artefacts d'extraction.
5. **Sauvegarder** chaque chunk dans `knowledge_base/rag/chunks/<section_id>.md` pour
   inspection humaine avant indexation.
6. **Indexer** dans ChromaDB avec les métadonnées. Si un chunk existe déjà pour la même
   clé `(source, section_id, chunk_index)`, le mettre à jour plutôt que de le dupliquer.
7. **Afficher** un rapport de synthèse : nombre de chunks par section, longueur moyenne,
   distance aux autres chunks (pour vérifier qu'ils sont bien distincts).

## Modèle d'embeddings

Utiliser le **même modèle d'embeddings** que celui déjà configuré pour ChromaDB dans le
projet. Si ce n'est pas clair, vérifier dans `agents/report/pipeline/_03_completion_plan.py`
et dans les tools `tools/build_pdf/search_exemplars.py`. **Ne pas en choisir un nouveau sans
cohérence** avec la recherche existante.

## Après indexation — Vérifications

Une fois l'indexation terminée, Claude Code doit vérifier que :

1. La collection ChromaDB contient bien les 8 chunks attendus (7 sections + 1 guide de style).
2. Une requête test par section_id retourne bien le bon chunk avec une distance faible (< 0.5).
3. Le module `search_exemplars.run()` utilisé par l'étape 03 du pipeline retourne maintenant
   des résultats non vides.

## Correction complémentaire — Augmenter la limite de troncature

Une fois l'indexation fonctionnelle, modifier `agents/report/pipeline/_03_completion_plan.py` :

- Passer la limite de troncature de **600 à 1800 caractères**.
- Passer le nombre de chunks récupérés par section de **1 à 3** (un chunk de contenu +
  si possible le guide de style systématiquement injecté).

## Livrable attendu

1. Fichier `knowledge_base/rag/build_rag.py` testé et fonctionnel.
2. Dossier `knowledge_base/rag/chunks/` avec les 8 fichiers markdown.
3. Base ChromaDB peuplée avec les 8 chunks.
4. Modification de `_03_completion_plan.py` pour la troncature à 1800 et 3 chunks par section.
5. Un petit rapport dans la réponse : nombre de chunks indexés, longueur moyenne,
   résultat d'une requête test sur `preamble`.

## Ne pas faire

- Ne pas générer les chunks avec un LLM (paraphrase) : on veut le texte **exact** du rapport
  de référence pour que le style soit transmis fidèlement.
- Ne pas chunker sur des longueurs fixes (500/1000 chars) qui couperaient au milieu de
  raisonnements actuariels. Chunker sur la structure sémantique (sections du rapport).
- Ne pas mélanger le guide de style avec les chunks de contenu : il doit être dans un
  chunk séparé avec `chunk_type="style_guide"`.
- Ne pas ignorer la cohérence avec ChromaDB existant (modèle d'embeddings, chemin,
  nom de collection) — c'est ce qui casserait l'intégration avec le pipeline.
