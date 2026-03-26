Introduction
============

.. admonition:: Ce que vous apprendrez dans cette page

   - Ce qu'est l'Agent Actuariel et à quoi il sert
   - À qui il s'adresse et ce qu'il produit
   - Vue d'ensemble du flux de travail, de l'import des données au rapport final

Qu'est-ce que l'Agent Actuariel ?
-----------------------------------

L'Agent Actuariel est un outil qui automatise la construction d'une **table de
mortalité d'expérience** à partir des données d'un portefeuille d'assurance.

Une table de mortalité d'expérience, c'est un tableau qui donne, pour chaque âge,
la probabilité de décès observée **dans votre portefeuille**. Elle est différente
des tables réglementaires publiées (TH 00-02, TF 00-02) parce qu'elle reflète
la réalité de vos assurés — leur santé, leurs comportements, leur sélection à
la souscription.

À qui s'adresse cet outil ?
------------------------------

L'outil est destiné aux **actuaires** en charge de la tarification ou du
provisionnement vie/prévoyance. Il suppose :

- un fichier de données d'assurés au format CSV (voir la section :ref:`configuration`)
- une connaissance de base des concepts actuariels (exposition, taux bruts, lissage)

.. note::

   Vous n'avez **pas besoin** de savoir programmer. L'agent écrit et exécute le
   code Python à votre place. Votre rôle est de valider ses choix méthodologiques
   et de lui fournir les bonnes données.

Ce que l'outil produit
------------------------

À la fin du processus, l'agent fournit :

1. **Une table de mortalité lissée** — fichier CSV exportable dans votre outil actuariel
2. **Des graphiques de validation** — courbe lissée vs taux bruts, bandes de confiance, courbe de survie
3. **Un SMR** (Standardized Mortality Ratio) — rapport entre mortalité observée et table de référence
4. **Une synthèse méthodologique** — justification de chaque choix (méthode de lissage, seuils utilisés)

Architecture générale
-----------------------

Le flux de traitement se déroule ainsi :

.. mermaid::

   flowchart LR
       A["Fichier CSV\n(données assurés)"] -->|"Chargement"| B["Interface Canvas\n(canvas_app.py)"]
       B -->|"Déclenchement"| C["Agent ReAct\n(agent.py)"]
       C -->|"Exécution code"| D["Notebooks actuariels\n(01→08)"]
       D -->|"Résultats"| E["État partagé\n(actuary_state.py)"]
       E -->|"Contexte RAG"| F["Chat avec les données\n(rag.py)"]
       C -->|"Table finale"| G["Rapport / Export CSV"]

       style A fill:#f9f,stroke:#333
       style G fill:#9f9,stroke:#333

.. note::

   Chaque flèche représente un transfert d'information. L'état partagé
   (``actuary_state.py``) est le «cerveau mémoire» du système : il permet à
   l'agent et au chat de partager les mêmes données sans se les envoyer en copie.

En pratique
-----------

Voici ce qui se passe quand vous cliquez sur «Lancer l'analyse» :

1. L'agent charge votre fichier CSV et nettoie les données.
2. Il calcule les expositions en années-personnes pour chaque âge.
3. Il calcule les taux bruts de mortalité.
4. Il diagnostique la crédibilité statistique de vos données.
5. Il teste plusieurs méthodes de lissage et choisit la meilleure (ou vous demande de trancher).
6. Il valide la table par intervalles de confiance et test du chi-deux.
7. Il compare votre portefeuille à la table de référence (SMR).
8. Il génère les graphiques et exporte la table finale.

.. warning::

   Le processus nécessite une connexion Internet active pour accéder à l'API
   OpenAI. Assurez-vous que la clé ``OPENAI_API_KEY`` est correctement configurée
   dans votre fichier ``.env`` avant de lancer une analyse.
