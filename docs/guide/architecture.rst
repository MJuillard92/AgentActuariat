Architecture du système
=======================

.. admonition:: Ce que vous apprendrez dans cette page

   - Les composants du système et leur rôle respectif
   - Comment ils s'articulent entre eux
   - Ce qui se passe «sous le capot» lors d'une analyse

Vue d'ensemble des composants
-------------------------------

Le système est composé de sept modules principaux :

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Fichier
     - Rôle
   * - ``canvas_app.py``
     - Interface utilisateur (Streamlit). Collecte les fichiers, affiche les résultats, gère le chat.
   * - ``agent.py``
     - Agent ReAct : boucle de raisonnement qui appelle le LLM et exécute le code.
   * - ``rag.py``
     - Moteur de chat avec les données (RAG). Répond aux questions sur les résultats.
   * - ``workflow_executor.py``
     - Exécuteur de workflow : lance les notebooks en ordre logique.
   * - ``notebooks/``
     - Bibliothèque actuarielle (01 à 08) : data_prep, exposure, crude_rates, etc.
   * - ``actuary_state.py``
     - État partagé (singleton) : kernel Python + cache d'embeddings.
   * - ``actuarial_params.py``
     - Configuration métier : seuils, bornes d'âge, paramètres de lissage.

Diagramme d'architecture
--------------------------

.. mermaid::

   flowchart TB
       subgraph UI["Interface utilisateur"]
           CA["canvas_app.py\n(Streamlit)"]
       end

       subgraph Cerveau["Moteur d'intelligence"]
           AG["agent.py\n(Boucle ReAct)"]
           RAG["rag.py\n(Chat avec les données)"]
       end

       subgraph Execution["Couche d'exécution"]
           WE["workflow_executor.py\n(Pipeline)"]
           NB["notebooks/\n01_data_prep … 08_visualization"]
       end

       subgraph Memoire["Mémoire partagée"]
           AS["actuary_state.py\n(Singleton)"]
           AP["actuarial_params.py\n(Configuration)"]
       end

       CA -->|"Lance l'analyse"| AG
       CA -->|"Question utilisateur"| RAG
       AG -->|"execute_python()"| WE
       WE -->|"exec() cellules"| NB
       NB -->|"DataFrames, résultats"| AS
       AS -->|"Kernel + embeddings"| RAG
       AP -->|"PARAMS (seuils)"| AG
       AP -->|"PARAMS (seuils)"| WE
       AG -->|"Résultats, graphiques"| CA
       RAG -->|"Réponse textuelle"| CA

       style CA fill:#4a90d9,color:#fff
       style AG fill:#e8a838,color:#fff
       style RAG fill:#e8a838,color:#fff
       style AS fill:#7bc47f,color:#fff
       style AP fill:#7bc47f,color:#fff

Rôle de chaque composant en détail
-------------------------------------

Interface utilisateur (canvas_app.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

C'est la porte d'entrée du système. Elle permet de :

- **Uploader** un fichier CSV de données assurés
- **Lancer** une analyse (mode agent libre) ou un workflow prédéfini
- **Visualiser** les résultats en temps réel (graphiques, logs d'étapes)
- **Poser des questions** sur les résultats via le chat RAG

Agent ReAct (agent.py)
~~~~~~~~~~~~~~~~~~~~~~~~

L'agent est le «chef de projet» qui orchestre l'analyse. Il reçoit un message
de l'utilisateur, réfléchit, et décide quelles lignes de code exécuter pour
avancer vers l'objectif.

Voir la page :doc:`agent` pour le détail de son fonctionnement.

Moteur RAG (rag.py)
~~~~~~~~~~~~~~~~~~~~~

Le RAG (Retrieval-Augmented Generation) permet de «discuter» avec les résultats
d'une analyse terminée. Il retrouve les passages pertinents des logs pour répondre
aux questions sans avoir à relancer les calculs.

Voir la page :doc:`rag` pour le détail.

Exécuteur de workflow (workflow_executor.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quand l'analyse est lancée en mode «workflow» (séquence prédéfinie plutôt que
agent libre), cet exécuteur parcourt le graphe de nœuds et exécute les notebooks
dans l'ordre, en évaluant les conditions des arêtes.

Voir la page :doc:`workflow` pour configurer un workflow.

Bibliothèque actuarielle (notebooks/)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Huit modules Python contenant toutes les fonctions actuarielles :

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Module
     - Fonctions principales
   * - ``01_data_preparation``
     - ``load_data``, ``clean_data``, ``compute_ages``, ``detect_anomalies``
   * - ``02_exposure``
     - ``compute_exposure_by_age``, ``exposure_summary``
   * - ``03_crude_rates``
     - ``crude_rates_central``, ``crude_rates_binomial``, ``crude_rates_kaplan_meier``
   * - ``04_smoothing``
     - ``smooth_whittaker``, ``smooth_gompertz``, ``smooth_makeham``, ``smooth_spline``
   * - ``05_diagnostics``
     - ``diagnose_credibility``, ``diagnose_monotonicity``, ``compare_smoothers``, ``compute_smr``
   * - ``06_validation``
     - ``confidence_intervals``, ``chi_square_test``, ``prudence_margin``, ``cox_model``
   * - ``07_benchmarking``
     - ``load_reference_table``, ``abatement_factors``, ``export_table``
   * - ``08_visualization``
     - ``plot_exposure_by_age``, ``plot_crude_vs_smoothed``, ``plot_confidence_bands``

État partagé (actuary_state.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Un singleton thread-safe qui centralise le kernel Python (espace de noms avec
tous les DataFrames et modules) et le cache d'embeddings pour le RAG.

Sans cet objet partagé, l'agent et le chat RAG auraient des visions différentes
des données — par exemple, le RAG ne saurait pas qu'un nouveau lissage a été
calculé entre deux questions.

Configuration (actuarial_params.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tous les paramètres métier sont dans ce fichier. Vous pouvez les modifier sans
toucher au code. Voir la page :doc:`configuration` pour le détail.

.. warning::

   Après toute modification de ``actuarial_params.py``, redémarrez l'application.
   Les paramètres sont chargés **une seule fois** au démarrage.
