Sélection automatique du modèle de lissage
===========================================

.. admonition:: Ce que vous apprendrez dans cette page

   - Pourquoi les taux bruts doivent être lissés
   - Comment fonctionne la sélection automatique ``auto_select_smoother``
   - Les quatre méthodes de lissage disponibles et leurs différences
   - Comment interpréter les trois statuts possibles (clear / close / escalate)

Pourquoi lisser les taux bruts ?
----------------------------------

Les taux bruts sont calculés directement à partir des données : ``q_x = D_x / E_x``.
Ils sont **non biaisés** mais **très bruités** : une année avec 3 décès de plus ou
de moins au même âge produit un saut dans la courbe qui n'est pas réel mais purement
statistique.

Pour produire une table utilisable en tarification et provisionnement, il faut une
courbe **régulière** (montée continue avec l'âge) et **robuste** (pas trop sensible
aux fluctuations d'une seule année).

C'est le rôle du lissage.

Diagramme de la sélection automatique
---------------------------------------

.. mermaid::

   flowchart TD
       A["Entrée : df_qx + df_exposure"] --> B["diagnose_credibility()\nQuel % des âges ont E_x < seuil ?"]
       B --> C{Niveau de crédibilité}
       C -->|"Non-paramétrique\n(données denses)"| D["Candidats :\nWhittaker + Spline"]
       C -->|"Mixte"| E["Candidats :\nWhittaker + Gompertz"]
       C -->|"Paramétrique\n(données peu denses)"| F["Candidats :\nGompertz + Makeham"]
       D --> G["Exécuter chaque modèle candidat"]
       E --> G
       F --> G
       G --> H["compare_smoothers()\nAIC Poisson pour chaque modèle"]
       H --> I{Meilleur modèle :\nviolations de monotonicité ?}
       I -->|"violations > mono_violations_max"| J["status = 'escalate'\nIntervention requise"]
       I -->|"OK"| K{Écart AIC entre\n1er et 2e modèle}
       K -->|"écart < aic_gap_threshold\n(modèles trop proches)"| L["status = 'close'\nChoix soumis à l'actuaire"]
       K -->|"écart ≥ seuil\n(gagnant clair)"| M["status = 'clear'\nContinuer automatiquement"]

       style J fill:#e85c5c,color:#fff
       style L fill:#f0ad4e,color:#000
       style M fill:#7bc47f,color:#fff

Les quatre méthodes de lissage
---------------------------------

Whittaker-Henderson
~~~~~~~~~~~~~~~~~~~~~

**Analogie :** Imaginez les taux bruts comme des poteaux plantés en terre à des hauteurs
irrégulières. Whittaker-Henderson tend une corde souple entre ces poteaux : elle passe
près de chaque poteau (fidélité aux données) mais évite de monter-descendre trop brusquement
entre deux poteaux voisins (régularité). Le paramètre ``lambda_wh`` contrôle la rigidité
de la corde.

**Avantages :**
- Très flexible : s'adapte à presque toutes les formes de courbe
- Respecte les données denses aux âges centraux
- Performant sur les grands portefeuilles

**Inconvénients :**
- Peut produire des irrégularités aux âges extrêmes (jeunes et très vieux)
- Nécessite des données suffisamment denses

**Quand l'utiliser :** portefeuilles avec bonne couverture de toutes les tranches d'âge.

Gompertz
~~~~~~~~~

**Analogie :** Gompertz part d'une idée simple : la mortalité **double** à peu près tous
les 8-10 ans après 30 ans. En mathématiques, cela donne ``log(μ_x) = a + b·x`` — une droite
sur l'échelle logarithmique. On ajuste simplement les deux paramètres ``a`` et ``b`` pour que
cette droite corresponde au mieux à vos données.

**Avantages :**
- Très robuste aux faibles effectifs (2 paramètres seulement)
- Extrapolation cohérente aux grands âges
- Monotone par construction

**Inconvénients :**
- Trop rigide si la mortalité n'est pas Gompertzienne dans votre portefeuille
- Ne capture pas la «bosse accidentelle» des jeunes âges

**Quand l'utiliser :** données éparses (< 10 000 contrats), portefeuille senior (45+ ans).

Makeham
~~~~~~~~

**Analogie :** Makeham est une extension de Gompertz qui ajoute un «socle» de mortalité
constant à tout âge (accidents, maladies non liées à l'âge). La formule est
``μ_x = A + B·exp(c·x)`` où ``A`` représente ce socle. Cela donne une courbe en forme de «J»
au lieu d'une droite logarithmique.

**Avantages :**
- Capture la mortalité accidentelle des jeunes adultes
- Encore robuste avec peu de données (3 paramètres)

**Inconvénients :**
- Peut être difficile à ajuster si les données aux jeunes âges sont insuffisantes

**Quand l'utiliser :** portefeuilles incluant des jeunes adultes (18-40 ans) avec mortalité
accidentelle visible.

Spline
~~~~~~~

**Analogie :** Une spline est comme un dessin réalisé à la règle souple des anciens
architectes (une longue réglette de bois courbée) : on impose des points de passage et la
règle se courbe naturellement entre eux, garantissant une courbe lisse et continue.

**Avantages :**
- Très flexible, capture les formes complexes
- Excellente précision sur les données denses

**Inconvénients :**
- Plus susceptible de «sur-ajuster» les fluctuations si les données sont bruitées
- Comportement moins prévisible aux extrémités

**Quand l'utiliser :** grands portefeuilles avec données très denses sur toute la plage d'âge.

Interpréter les trois statuts
-------------------------------

``status = "clear"`` — Gagnant net
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Un modèle est clairement meilleur que les autres : écart AIC supérieur au seuil
(2 unités par défaut) et aucune violation de monotonicité après 40 ans.

**Action :** l'agent continue automatiquement avec ce modèle. Vous n'avez rien à faire.

Exemple de message :

.. code-block:: text

   ✓ Sélection du modèle : CLEAR
      Meilleur modèle  : whittaker
      Décision         : Modèle sélectionné : 'whittaker' (AIC=3241.2, monotonicité OK)
                         (2e : 'spline', AIC=3248.7). Crédibilité : non-parametric (8.3 % âges
                         à faible crédibilité).

``status = "close"`` — Deux modèles proches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

L'écart AIC entre le premier et le deuxième modèle est inférieur au seuil. Sur le plan
statistique, les deux modèles sont équivalents. Le choix final nécessite un **jugement
actuariel** (connaissance du portefeuille, prudence, cohérence avec les études passées).

**Action :** l'agent s'arrête et vous présente un tableau comparatif. Choisissez le
modèle en examinant les colonnes ``MSE_vs_crude`` (fidélité aux données) et
``n_non_monotone`` (violations de monotonicité).

.. note::

   En cas de doute, préférez le modèle paramétrique (Gompertz/Makeham) : il sera
   plus cohérent avec les tables de référence et plus défendable lors d'une revue
   réglementaire.

``status = "escalate"`` — Intervention requise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le meilleur modèle présente des violations de monotonicité inacceptables (au moins 1
âge x tel que ``q_x > q_{x+1}`` après 40 ans), **ou** aucun modèle n'a convergé.

**Action :** plusieurs pistes à explorer avant de relancer :

1. **Augmenter ``lambda_wh``** dans ``actuarial_params.py`` (p. ex. de 100 à 200) pour un lissage plus fort.
2. **Restreindre la plage d'âges** (``age_max``) si les grands âges ont très peu d'observations.
3. **Forcer un modèle paramétrique** en retirant ``"whittaker"`` et ``"spline"`` des candidats.
4. **Vérifier les données** : une non-monotonicité persistante après lissage peut indiquer une anomalie dans le fichier source (assurés sortants mal datés, etc.).

.. warning::

   Ne jamais ignorer une escalade en passant manuellement au modèle suivant sans avoir
   compris la cause. Une table non monotone peut produire des primes incorrectes et des
   provisions insuffisantes.

Critère de comparaison : l'AIC Poisson
-----------------------------------------

L'AIC Poisson est le critère principal utilisé pour classer les modèles. Il mesure
le compromis entre **fidélité aux données** (vraisemblance Poisson) et **complexité**
(nombre de paramètres).

La formule est : ``AIC = -2 × log-vraisemblance + 2 × nombre_de_paramètres``

Un AIC plus **bas** est meilleur. L'AIC Poisson est préféré au MSE car il prend en
compte l'hétéroscédasticité des taux de mortalité (les taux aux grands âges sont plus
variables que ceux aux âges centraux).
