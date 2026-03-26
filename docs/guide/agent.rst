L'agent ReAct
=============

.. admonition:: Ce que vous apprendrez dans cette page

   - Ce qu'est le patron ReAct et pourquoi il est utilisé
   - Le flux exact d'une analyse complète (10 étapes)
   - Les deux outils disponibles et quand l'agent les utilise
   - Comment interpréter les messages affichés pendant l'analyse

Qu'est-ce que le patron ReAct ?
---------------------------------

ReAct (Reasoning + Acting) est une façon de faire travailler un LLM en mode
«essai-erreur» plutôt qu'en mode «réponse unique» :

1. Le LLM **réfléchit** : il observe la situation et décide quoi faire.
2. Le LLM **agit** : il appelle un outil (exécute du code, cherche de la documentation).
3. Le LLM **observe** : il reçoit le résultat de l'outil.
4. On recommence jusqu'à ce que la tâche soit terminée.

Cette boucle est particulièrement adaptée aux analyses actuarielles, car chaque
étape produit des résultats qui informent les décisions suivantes : le diagnostic
de crédibilité (étape 4) détermine quelle méthode de lissage choisir (étape 5).

Diagramme de la boucle ReAct
------------------------------

.. mermaid::

   flowchart TD
       A["Message utilisateur\n(ex: 'Analyse ce fichier CSV')"] --> B["LLM : raisonnement\n(choisit quoi faire)"]
       B --> C{Appel d'outil ?}
       C -->|"Oui"| D{Quel outil ?}
       D -->|"execute_python"| E["Exécution du code\ndans le kernel"]
       D -->|"search_documentation"| F["Recherche dans la\nbase documentaire"]
       E --> G["Résultat renvoyé\nau LLM"]
       F --> G
       G --> B
       C -->|"Non (finish_reason = stop)"| H["Synthèse finale\n(résumé méthodologique)"]
       H --> I["Fin de l'analyse"]
       B -->|"MAX_ITERATIONS atteint"| J["Erreur : limite dépassée"]

       style A fill:#4a90d9,color:#fff
       style H fill:#7bc47f,color:#fff
       style I fill:#7bc47f,color:#fff
       style J fill:#e85c5c,color:#fff

Les deux outils disponibles
-----------------------------

``execute_python`` — Exécuter du code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

C'est l'outil principal. L'agent écrit du code Python et le fait exécuter dans
un environnement partagé («kernel») qui persiste tout au long de l'analyse.

**Ce que cela signifie en pratique :**

- Les variables calculées à l'étape 2 (expositions) sont encore disponibles à l'étape 7 (SMR).
- L'agent peut corriger son code si une erreur se produit et relancer.
- Vous voyez en temps réel ce que l'agent calcule.

.. note::

   Chaque appel ``execute_python`` affiche une **description en français** à la place
   du code brut. Par exemple : «Calcul des taux bruts par la méthode centrale».
   Si vous souhaitez voir le code, développez le panneau «Code Python» dans l'interface.

``search_documentation`` — Chercher de la documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cet outil permet à l'agent de consulter votre base documentaire actuarielle (notes
méthodologiques passées, décisions de comité technique, justifications de choix).

L'agent est instruit de l'utiliser **avant toute décision de jugement** :
- Si la sélection automatique retourne «close» (deux modèles proches)
- Si le SMR est anormal (hors de la plage 0.80–1.20)
- Pour justifier ses choix dans le rapport final

.. note::

   La connexion à la base documentaire sera activée dans une version ultérieure.
   En attendant, l'agent procède avec son jugement d'expert actuariel intégré.

Le processus standard en 10 étapes
-------------------------------------

Quand vous demandez à l'agent de construire une table de mortalité, il suit ce
processus standard (sauf exception méthodologique) :

**Étape 1 — Préparation des données**

.. code-block:: text

   data_prep.load_data() → clean_data() → compute_ages() → detect_anomalies()

Charge le fichier CSV, valide la cohérence des dates, calcule les âges, détecte
les anomalies structurelles (doublons, données manquantes, incohérences de dates).

**Étape 2 — Calcul des expositions**

.. code-block:: text

   exposure.compute_exposure_by_age() → exposure_summary()

Pour chaque âge entier, calcule l'exposition en années-personnes (E_x) et le
nombre de décès observés (D_x).

**Étape 3 — Taux bruts de mortalité**

.. code-block:: text

   crude_rates.crude_rates_central()  ← méthode par défaut
   crude_rates.crude_rates_kaplan_meier()  ← si n < 5 000 contrats

Calcule les taux bruts q_x = D_x / E_x. Pour les petits portefeuilles, utilise
l'estimateur Kaplan-Meier qui gère mieux la censure.

**Étape 4 — Diagnostic de crédibilité (OBLIGATOIRE)**

.. code-block:: text

   diagnostics.diagnose_credibility()

Mesure le pourcentage d'âges avec moins de ``threshold_low`` années-personnes
d'exposition. Le résultat oriente le choix du modèle de lissage.

**Étape 5 — Sélection automatique du lissage (OBLIGATOIRE)**

.. code-block:: text

   smoothing_selector.auto_select_smoother()
   → status: "clear" / "close" / "escalate"

L'étape la plus importante. Voir la page :doc:`smoothing` pour les détails.

.. warning::

   Si l'agent retourne ``status = "close"`` ou ``"escalate"``, il **s'arrête**
   et attend votre intervention. Ne relancez pas l'analyse sans avoir examiné
   les résultats affichés.

**Étape 6 — Validation statistique**

.. code-block:: text

   validation.confidence_intervals() + chi_square_test() + prudence_margin()

Calcule les intervalles de confiance à 95 %, teste l'adéquation observé/attendu
et vérifie que la table est suffisamment prudente.

**Étape 7 — SMR et benchmarking**

.. code-block:: text

   diagnostics.compute_smr() + benchmarking.abatement_factors()

Compare votre portefeuille à la table de référence TH/TF 00-02. Calcule les
facteurs d'abattement âge par âge.

**Étape 8 — Visualisations**

.. code-block:: text

   visualization.plot_crude_vs_smoothed()
   visualization.plot_confidence_bands()
   visualization.plot_survival_curve()
   ...

Génère les graphiques standard. Tous sont capturés automatiquement et affichés
dans l'interface.

**Étape 9 — Export de la table**

.. code-block:: text

   benchmarking.export_table()

Exporte la table finale au format CSV dans le dossier ``outputs/``.

**Étape 10 — Synthèse méthodologique**

L'agent rédige un résumé expliquant chaque choix : méthode de taux bruts retenue,
modèle de lissage sélectionné et pourquoi, résultat du SMR, interprétation des
tests statistiques.

Garde-fous intégrés
--------------------

L'agent comporte plusieurs mécanismes de protection :

- **Limite d'itérations** : maximum 25 appels d'outils par session (évite les boucles infinies).
- **Troncature des sorties** : les logs trop longs sont coupés à 3 000 caractères.
- **Erreurs corrigées automatiquement** : si un appel ``execute_python`` produit une erreur Python, le message d'erreur est renvoyé au LLM qui corrige et relance.
- **Paramètres depuis PARAMS** : l'agent est instruit de ne jamais hardcoder de valeurs numériques — il lit toujours depuis ``PARAMS``.

En pratique
-----------

Voici un exemple d'échange typique :

.. code-block:: text

   Vous    : Analyse le fichier data/portefeuille_H_2023.csv et construis la table.

   Agent   : [Étape 1] Chargement et nettoyage des données...
             → 47 832 contrats chargés, 3 doublons supprimés, 0 valeur manquante critique.

   Agent   : [Étape 2] Calcul des expositions par âge...
             → Exposition totale : 112 456 années-personnes, 1 234 décès observés.
             → 8.3 % des âges avec exposition < 10 ans-personnes.

   Agent   : [Étape 4] Diagnostic de crédibilité...
             → Recommandation : "non-parametric" (8.3 % < seuil 10 %)

   Agent   : [Étape 5] Sélection automatique du lissage...
             → Status : "clear" — Whittaker-Henderson retenu (AIC=3241.2 vs Spline=3248.7)

   Agent   : [Étapes 6-9] Validation, SMR, graphiques, export...
             → SMR = 0.94 (IC 95 % : [0.89, 0.99]) — sélection favorable légère.

   Agent   : Synthèse : Table construite avec Whittaker-Henderson (λ=100, d=2).
             Portefeuille légèrement sous la mortalité de référence (SMR=0.94).
             Chi-deux : p=0.23 (adéquation satisfaisante). Table exportée.
