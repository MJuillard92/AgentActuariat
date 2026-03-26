Chat avec les données (RAG)
============================

.. admonition:: Ce que vous apprendrez dans cette page

   - Ce qu'est le RAG et à quoi il sert dans ce contexte
   - Comment poser de bonnes questions au système
   - Les deux modes de chat disponibles (classique et avec outils)
   - Les outils que le LLM peut utiliser pour approfondir une réponse

Qu'est-ce que le RAG ?
-----------------------

RAG signifie **Retrieval-Augmented Generation** : génération augmentée par récupération.

En termes simples : plutôt que de demander au LLM ce qu'il sait «par cœur», on
lui fournit les passages pertinents des **résultats de votre analyse** et on lui
demande de répondre à partir de ces extraits.

**Exemple :**

Vous posez la question : «Quel est le SMR pour les hommes de 65 à 70 ans ?»

Le système :

1. Calcule la «distance» entre votre question et chaque étape de l'analyse
2. Sélectionne les 5-10 étapes les plus proches (celles qui parlent de SMR)
3. Injecte ces extraits dans le contexte du LLM
4. Demande au LLM de répondre **uniquement** à partir de ces extraits

Diagramme du pipeline RAG
---------------------------

.. mermaid::

   flowchart LR
       A["Question utilisateur\n(texte)"] --> B["Embedding de la question\n(text-embedding-3-small)"]
       B --> C["Similarité cosinus\navec les chunks en cache"]
       C --> D["Sélection des top-k\nchunks pertinents"]
       D --> E["Construction du prompt\n(contexte + question)"]
       E --> F["LLM (GPT-4o-mini)"]

       subgraph Cache["Cache d'embeddings (ActuaryState)"]
           G["Chunks des étapes\n(logs, résultats)"]
           H["Embeddings pré-calculés"]
           G --- H
       end

       H --> C

       F --> I{Besoin d'un outil ?}
       I -->|"Oui"| J{Quel outil ?}
       J -->|"execute_python"| K["Calcul dans le kernel\n(accès aux DataFrames)"]
       J -->|"list_available_data"| L["Liste des variables\ndisponibles"]
       J -->|"get_dataframe_info"| M["Détail d'un DataFrame\n(colonnes, stats, aperçu)"]
       K --> F
       L --> F
       M --> F
       I -->|"Non"| N["Réponse finale\n(texte + graphiques éventuels)"]

       style N fill:#7bc47f,color:#fff

Le cache d'embeddings
-----------------------

Dès que l'analyse est terminée, les embeddings de toutes les étapes sont
**pré-calculés une seule fois** et mis en cache dans l'état partagé (``ActuaryState``).

Cela signifie que chaque question coûte un seul petit appel API (l'embedding de
votre question), et non un appel pour chaque étape de l'analyse. Les réponses
sont donc rapides même si l'analyse comporte de nombreuses étapes.

Le cache est automatiquement invalidé quand une nouvelle analyse est lancée.

Les deux modes de chat
------------------------

Mode classique (``answer_with_rag``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le LLM répond uniquement à partir des extraits retrouvés. C'est le mode par défaut.

**Adapté pour :**

- Questions sur les résultats numériques déjà calculés (SMR, AIC, intervalles de confiance)
- Questions d'interprétation («Que signifie un SMR de 0.94 ?»)
- Questions sur les choix méthodologiques effectués

Mode avec outils (``answer_with_tools``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

En plus des extraits RAG, le LLM peut exécuter du code Python pour calculer
des résultats non présents dans les logs.

**Adapté pour :**

- Requêtes sur des sous-populations («Quel est le taux de mortalité des femmes de 55-60 ans ?»)
- Recalculs à la demande («Recalcule le SMR en excluant les âges < 45 ans»)
- Génération de graphiques supplémentaires («Trace la courbe de survie pour les 65-80 ans»)
- Exploration des données supprimées lors du nettoyage

Les trois outils disponibles en mode «avec outils»
-----------------------------------------------------

``execute_python`` — Exécuter du code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Exécute du code Python dans le kernel de l'analyse. Le LLM a accès à tous les
DataFrames calculés : ``df``, ``df_clean``, ``df_exposure``, ``df_qx``, ``df_smooth``,
ainsi qu'à tous les modules actuariels.

.. note::

   Le LLM est instruit d'utiliser ``print()`` pour afficher les résultats.
   Un graphique matplotlib est automatiquement capturé et affiché dans l'interface.

``list_available_data`` — Lister les données
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retourne la liste complète des variables disponibles dans le kernel, avec leur
type et leur taille. Le LLM doit appeler cet outil **en premier** avant d'accéder
à une variable, pour éviter d'utiliser un nom qui n'existe pas.

Exemple de réponse :

.. code-block:: text

   === Objets disponibles dans le namespace de l'analyse ===
     • df_exposure: DataFrame 71×4 — colonnes: ['age', 'E_x', 'D_x', 'q_x_brut']
     • df_qx: DataFrame 71×5 — colonnes: ['age', 'E_x', 'D_x', 'q_x_brut', 'qx']
     • df_smooth: DataFrame 71×3 — colonnes: ['age', 'qx_smoothed', 'method']
     • PARAMS = {'observation': {...}, 'ages': {...}, ...}

``get_dataframe_info`` — Détail d'un DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retourne les informations détaillées d'un DataFrame spécifique : forme, noms
des colonnes, statistiques descriptives, premières lignes, valeurs manquantes.

Le LLM utilise cet outil avant d'écrire du code pour s'assurer d'utiliser les
bons noms de colonnes.

Poser de bonnes questions
---------------------------

Pour obtenir les meilleures réponses, formulez vos questions de manière précise :

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Question vague
     - Question précise
   * - «Comment va la mortalité ?»
     - «Quel est le SMR global et son intervalle de confiance à 95 % ?»
   * - «Les données sont-elles bonnes ?»
     - «Combien de contrats ont été supprimés lors du nettoyage et pour quelle raison ?»
   * - «Le lissage est-il correct ?»
     - «Combien de violations de monotonicité le modèle Whittaker-Henderson présente-t-il après 40 ans ?»
   * - «Montre-moi les jeunes»
     - «Trace le graphique des taux bruts pour les âges 20 à 40 ans»

.. warning::

   Le RAG ne peut répondre qu'aux questions portant sur les résultats de la
   **dernière analyse lancée**. Si vous rechargez de nouvelles données, les réponses
   du RAG porteront sur la nouvelle analyse dès que le cache est reconstruit.

Fenêtre contextuelle dynamique
--------------------------------

Le système calcule automatiquement combien d'extraits injecter dans le contexte,
selon la taille de votre question et la longueur des extraits disponibles, sans
jamais dépasser la fenêtre maximale du modèle (128 000 tokens).

Cela garantit que les réponses sont toujours aussi complètes que possible, sans
erreur de dépassement de contexte.
