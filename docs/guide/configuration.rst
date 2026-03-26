.. _configuration:

Configuration des paramètres actuariels
=========================================

.. admonition:: Ce que vous apprendrez dans cette page

   - À quoi sert chaque paramètre du fichier ``actuarial_params.py``
   - Ce qui se passe si vous modifiez une valeur
   - Les valeurs typiques pour une étude de mortalité en France
   - Des exemples concrets selon votre type de portefeuille

Le fichier de configuration
-----------------------------

Tous les paramètres métier se trouvent dans le fichier ``actuarial_params.py``,
à la racine du projet. Ce fichier est conçu pour être modifié par un actuaire
**sans toucher au code** du reste de l'application.

.. warning::

   Après toute modification de ce fichier, vous devez **redémarrer l'application**
   (``Ctrl+C`` puis relancer ``streamlit run canvas_app.py``). Les paramètres sont
   lus une seule fois au démarrage.

Groupe 1 : Période d'observation
----------------------------------

.. code-block:: python

   "observation": {
       "date_fin": "2023-12-31",
   }

**``date_fin``** — Fin de la période d'étude (format ``YYYY-MM-DD``).

Cette date sert de borne supérieure pour le calcul des expositions. Un assuré
encore en vie au-delà de cette date est considéré comme «sortant censurément»
le dernier jour de la période.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Scénario
     - Valeur recommandée
   * - Étude annuelle sur 2023
     - ``"2023-12-31"``
   * - Étude quinquennale 2019-2023
     - ``"2023-12-31"``
   * - Étude sur données partielles (T1 2024)
     - ``"2024-03-31"``

Groupe 2 : Plages d'âges
--------------------------

.. code-block:: python

   "ages": {
       "age_min": 20,
       "age_max": 90,
   }

**``age_min``** — Âge minimum retenu dans les calculs.

Les assurés plus jeunes sont ignorés. En France, l'essentiel de la mortalité
d'expérience utile commence vers 20-25 ans pour les produits de prévoyance
individuelle.

**``age_max``** — Âge maximum retenu.

Au-delà de cet âge, les effectifs sont généralement trop faibles pour produire
des taux fiables. La table est extrapolée ou tronquée.

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Type de portefeuille
     - ``age_min``
     - ``age_max``
   * - Assurance vie / rentes (adultes actifs)
     - 20
     - 90
   * - Portefeuille senior (retraite)
     - 55
     - 100
   * - Prévoyance collective entreprise
     - 18
     - 70
   * - Portefeuille jeunes (20-50 ans)
     - 20
     - 75

.. note::

   **Exemple concret :** si votre portefeuille ne contient presque aucun assuré
   de moins de 40 ans, positionner ``age_min: 40`` éliminera du bruit statistique
   et améliorera la qualité du lissage aux âges réellement représentatifs.

Groupe 3 : Lissage
-------------------

.. code-block:: python

   "smoothing": {
       "lambda_wh": 100,
       "d": 2,
       "gompertz_age_min": 40,
       "gompertz_age_max": 90,
       "makeham_age_min": 30,
       "makeham_age_max": 90,
       "local_poly_bandwidth": 5,
       "local_poly_degree": 2,
   }

Paramètres Whittaker-Henderson
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**``lambda_wh``** — Intensité du lissage (pénalité de rugosité).

C'est le paramètre le plus important du lissage. Imaginez une courbe tendue
entre les points observés : ``lambda_wh`` contrôle la «rigidité» de cette corde.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Valeur
     - Effet
   * - 10 — 50
     - Lissage faible : la courbe suit de près les points observés. Risque de «zigzags».
   * - 100 (défaut)
     - Équilibre entre fidélité aux données et régularité. Adapté à la plupart des portefeuilles.
   * - 200 — 500
     - Lissage fort : courbe très régulière. Utile si les effectifs sont faibles et les taux très bruités.

**``d``** — Ordre de la pénalité de différence.

- ``d = 2`` : pénalise les variations brusques de la courbe (courbure). Valeur standard.
- ``d = 3`` : pénalise les variations de la courbure. Produit des courbes encore plus douces.

Paramètres Gompertz
~~~~~~~~~~~~~~~~~~~~~

**``gompertz_age_min``** et **``gompertz_age_max``** — Âges utilisés pour l'ajustement
du modèle Gompertz (``log(μ_x) = a + b·x``).

En dessous de 40 ans, la mortalité ne suit pas bien la loi de Gompertz (présence de la
«bosse accidentelle»). Au-dessus de 90 ans, les données sont trop rares pour contraindre
l'ajustement.

Paramètres Makeham
~~~~~~~~~~~~~~~~~~~~

**``makeham_age_min``** — Doit être plus bas que Gompertz (30 ans par défaut) pour capturer
la bosse accidentelle des jeunes âges, caractéristique du modèle Makeham.

Paramètres polynôme local (LOESS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**``local_poly_bandwidth``** — Demi-fenêtre en nombre d'âges. Avec ``bandwidth = 5``, la
courbe au point x=60 est calculée à partir des observations de x=55 à x=65.

**``local_poly_degree``** — Degré du polynôme ajusté localement.
- ``1`` : linéaire (plus rapide, moins flexible)
- ``2`` : quadratique (standard, capture mieux les inflexions)

Groupe 4 : Crédibilité
-----------------------

.. code-block:: python

   "credibility": {
       "threshold_low": 10,
       "pct_parametric": 30,
       "pct_mixed": 10,
   }

**``threshold_low``** — Exposition en dessous de laquelle un âge est considéré
«à faible crédibilité» (en années-personnes).

Un âge x avec seulement 8 années-personnes d'exposition produit un taux brut très
incertain (variance élevée). Le seuil de 10 est une valeur standard en pratique française.

**``pct_parametric``** — Si plus de X % des âges ont une exposition faible, on force
un modèle paramétrique (Gompertz/Makeham). Ces modèles «empruntent de la force» aux
autres âges grâce à leur structure mathématique.

**``pct_mixed``** — Seuil intermédiaire : entre ``pct_mixed`` % et ``pct_parametric`` %
d'âges à faible crédibilité, on utilise une approche mixte.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Situation
     - Recommandation
   * - Grand portefeuille (> 100 000 contrats)
     - ``threshold_low: 5``, méthodes non-paramétriques suffisent
   * - Portefeuille moyen (10 000–100 000)
     - ``threshold_low: 10`` (valeurs par défaut)
   * - Petit portefeuille (< 10 000 contrats)
     - ``threshold_low: 20``, forcer le parametric si > 20 %

Groupe 5 : SMR
---------------

.. code-block:: python

   "smr": {
       "lower": 0.90,
       "upper": 1.10,
   }

Le SMR (Standardized Mortality Ratio) compare la mortalité de votre portefeuille
à une table de référence (TH 00-02 pour les hommes, TF 00-02 pour les femmes).

- **SMR = 1.0** : votre portefeuille a exactement la même mortalité que la référence.
- **SMR = 0.85** : vos assurés sont 15 % moins susceptibles de décéder (sélection favorable).
- **SMR = 1.20** : vos assurés sont 20 % plus susceptibles de décéder (surmortalité).

**``lower``** et **``upper``** définissent la zone «normale». En dehors de cette zone,
l'agent génère une alerte et vous recommande d'investiguer.

.. note::

   Un SMR hors de la plage [0.3, 3.0] signale probablement une **anomalie dans les données**
   (période d'observation incorrecte, biais de sélection majeur). L'agent vous préviendra
   automatiquement dans ce cas.

Groupe 6 : Validation statistique
------------------------------------

.. code-block:: python

   "validation": {
       "alpha": 0.05,
       "prudence_margin_min": 0.10,
       "chi_square_min_expected": 1.0,
   }

**``alpha``** — Niveau de signification des tests statistiques.

Avec ``alpha = 0.05``, les intervalles de confiance sont à 95 %. Valeur standard
en actuariat.

**``prudence_margin_min``** — Pour les produits d'assurance-vie/rentes, la table
doit être suffisamment prudente. Une marge de 10 % signifie que les taux lissés
doivent être au moins 10 % supérieurs aux taux observés bruts.

.. warning::

   Ne réduisez pas ``prudence_margin_min`` en dessous de 0 si la table est destinée
   à un provisionnement réglementaire. Une marge négative produirait une table
   «optimiste» non conforme aux exigences Solvabilité II.

**``chi_square_min_expected``** — Nombre attendu de décès minimum pour inclure
une cellule dans le test du chi-deux. Les cellules avec trop peu de décès attendus
rendent le test invalide (hypothèse d'approximation normale non vérifiée).

Groupe 7 : Diagnostics
-----------------------

.. code-block:: python

   "diagnostics": {
       "age_start_monotonicity": 40,
   }

**``age_start_monotonicity``** — Âge à partir duquel la monotonicité de la table
est vérifiée.

Avant 40 ans, il est normal que les taux de mortalité ne soient pas strictement
croissants avec l'âge («bosse accidentelle» liée aux accidents, suicides, etc.).
À partir de 40 ans, la mortalité biologique reprend le dessus et doit augmenter
monotonement avec l'âge.

Groupe 8 : Sélection automatique du modèle
--------------------------------------------

.. code-block:: python

   "model_selection": {
       "aic_gap_threshold": 2.0,
       "mono_violations_max": 0,
       "candidates_non_parametric": ["whittaker", "spline"],
       "candidates_mixed":          ["whittaker", "gompertz"],
       "candidates_parametric":     ["gompertz", "makeham"],
   }

**``aic_gap_threshold``** — Écart d'AIC Poisson en dessous duquel deux modèles
sont considérés «statistiquement équivalents».

La règle des 2 unités d'AIC est standard en sélection de modèles (Burnham &
Anderson, 2002). En dessous de ce seuil, le choix ne peut pas être fait par le
seul critère statistique — il faut un jugement actuariel.

**``mono_violations_max``** — Nombre maximal de violations de monotonicité
acceptées. Avec 0, le premier modèle non monotone après 40 ans déclenche une
escalade vers l'actuaire.

**``candidates_*``** — Listes de modèles à tester selon le niveau de crédibilité.
Vous pouvez retirer un modèle de la liste pour ne pas le tester (par exemple,
exclure ``"spline"`` si vous ne souhaitez jamais l'utiliser).
