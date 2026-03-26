Workflows Canvas
================

.. admonition:: Ce que vous apprendrez dans cette page

   - Ce qu'est un workflow et en quoi il diffère du mode agent libre
   - La structure d'un nœud et d'une arête
   - Comment configurer des conditions pour brancher le flux
   - Les cas d'usage typiques

Mode agent libre vs mode workflow
------------------------------------

L'outil propose deux façons de lancer une analyse :

**Mode agent libre** (``agent.py``)
  L'agent décide lui-même de l'ordre des étapes, des méthodes à utiliser et du
  nombre d'itérations. Adapté aux analyses exploratoires ou aux cas non standard.

**Mode workflow** (``workflow_executor.py``)
  Vous définissez à l'avance une séquence de nœuds connectés par des arêtes
  conditionnelles. L'exécuteur suit ce graphe de façon déterministe. Adapté aux
  analyses répétitives où vous voulez un résultat reproductible à chaque fois.

Qu'est-ce qu'un nœud ?
-----------------------

Un nœud (``WorkflowNode``) représente une **étape de traitement** associée à un
notebook actuariel. Quand l'exécuteur arrive sur un nœud, il exécute toutes les
cellules de code du notebook correspondant.

Propriétés d'un nœud :

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Propriété
     - Description
   * - ``id``
     - Identifiant unique du nœud (chaîne de caractères)
   * - ``label``
     - Nom affiché dans l'interface («Calcul des expositions»)
   * - ``notebook_path``
     - Chemin vers le fichier Python du notebook à exécuter

Qu'est-ce qu'une arête ?
-------------------------

Une arête (``WorkflowEdge``) connecte deux nœuds et peut porter une **condition**.
La condition est une expression Python évaluée sur les variables disponibles dans le
kernel après l'exécution du nœud source.

Propriétés d'une arête :

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Propriété
     - Description
   * - ``source``
     - ``id`` du nœud de départ
   * - ``target``
     - ``id`` du nœud d'arrivée
   * - ``condition``
     - Expression Python (vide = toujours vrai)

Conditions disponibles
-----------------------

La condition est évaluée avec les variables scalaires du kernel comme contexte.
Voici les variables typiquement disponibles :

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Variable
     - Signification
   * - ``SMR``
     - SMR global calculé par ``compute_smr()``
   * - ``non_mono``
     - Nombre de violations de monotonicité
   * - ``n_vides``
     - Nombre de valeurs manquantes critiques
   * - ``pct_low_cred``
     - Pourcentage d'âges à faible crédibilité
   * - ``True``
     - Toujours vrai (arête inconditionnelle)

Exemples de conditions :

.. code-block:: python

   "SMR > 1.2"          # Surmortalité significative
   "SMR < 0.8"          # Forte sélection favorable
   "non_mono > 0"       # Violations de monotonicité présentes
   "pct_low_cred > 30"  # Données trop peu denses → forcer paramétrique
   "True"               # Toujours exécuter ce nœud

Logique d'exécution
---------------------

L'exécuteur parcourt le graphe dans l'ordre topologique (un nœud n'est exécuté
que si toutes ses dépendances ont été traitées).

Pour chaque nœud, la logique est :

1. Y a-t-il des arêtes entrantes ?

   - **Non** → c'est un nœud de départ, on l'exécute sans condition.
   - **Oui** → on vérifie les arêtes des nœuds **déjà exécutés** qui pointent vers ce nœud.

2. Parmi ces arêtes actives, **au moins une** condition est-elle vraie ?

   - **Oui** → le nœud est exécuté.
   - **Non** → le nœud est ignoré (``skipped: True``).

3. Si une erreur se produit dans un nœud (``❌ Erreur``), l'exécution s'arrête.

Exemple de workflow conditionnel
----------------------------------

Ce workflow bifurque selon le SMR calculé à l'étape 5 :

.. mermaid::

   flowchart TD
       A["Nœud 1\nPréparation données"] --> B["Nœud 2\nExpositions"]
       B --> C["Nœud 3\nTaux bruts"]
       C --> D["Nœud 4\nLissage + SMR"]
       D -->|"SMR > 1.2"| E["Nœud 5a\nAlerte surmortalité\n(rapport spécial)"]
       D -->|"SMR <= 1.2"| F["Nœud 5b\nFlux standard\n(rapport normal)"]
       E --> G["Nœud 6\nExport final"]
       F --> G

.. note::

   Un nœud peut avoir plusieurs arêtes entrantes. Si les sources de deux arêtes
   entrantes sont toutes deux «ignorées» (``skipped``), le nœud est ignoré lui aussi.
   Cela garantit que les branches inutilisées ne s'accumulent pas en fin de graphe.

Créer un workflow dans l'interface
-------------------------------------

Dans l'onglet «Canvas» de l'interface :

1. **Ajouter un nœud** : cliquer sur «+ Nœud», renseigner le label et choisir le notebook.
2. **Connecter deux nœuds** : glisser depuis le port de sortie d'un nœud vers le port d'entrée d'un autre.
3. **Ajouter une condition** : cliquer sur une arête et saisir l'expression Python.
4. **Sauvegarder** : les workflows sont stockés dans le dossier ``workflows/``.
5. **Exécuter** : cliquer sur «Lancer le workflow».

.. warning::

   Les conditions des arêtes sont évaluées par ``eval()`` sur les variables du kernel.
   N'utilisez que des expressions simples (comparaisons de scalaires). N'écrivez pas
   de code qui appelle des fonctions ou importe des modules dans une condition.
