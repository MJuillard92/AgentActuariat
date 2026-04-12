# Exemple de niveau de détail attendu pour la description d'un modèle
#
# Cet exemple est injecté dans le prompt système du rédacteur LLM
# pour calibrer le niveau de formalisme mathématique attendu dans le rapport.
# Il est intentionnellement générique (pas spécifique mortalité) pour
# servir aussi bien pour les tables de mortalité que le provisionnement non-vie.
#
# RÈGLE DE NOTATION : indices et exposants en Unicode — jamais d'underscores.
#   ✓  q̂ₓ = Dₓ / Eₓ
#   ✗  q̂_x = D_x / E_x   ← INTERDIT

---

## Exemple : Estimateur de Kaplan-Meier central (exposition au risque)

### Contexte
Pour estimer un taux de mortalité (ou toute intensité de transition) à l'âge x, il faut
d'abord mesurer la durée totale passée sous risque à cet âge — appelée **exposition centrale**
Eₓ. L'estimateur de Kaplan-Meier central est l'approche de référence lorsque les données sont
individuelles et les durées d'observation partielles (censures à gauche et à droite).

### Formulation mathématique

L'exposition centrale à l'âge x est définie par :

    Eₓ = Σᵢ∈Rₓ [ min(Tᵢ⁺, x+1) − max(Tᵢ⁻, x) ]

où la somme porte sur l'ensemble des individus Rₓ présents sous risque à l'âge x, et :

| Symbole    | Définition                                                          |
|------------|---------------------------------------------------------------------|
| Tᵢ⁻        | Date d'entrée de l'individu i dans l'observation (fraction d'année) |
| Tᵢ⁺        | Date de sortie de l'individu i (décès, censure ou fin d'étude)      |
| x          | Âge en années entières (anniversaire)                               |
| min(·, x+1)| Troncature à la fin de l'année d'âge x                              |
| max(·, x)  | Troncature au début de l'année d'âge x                              |

Cette formule mesure la fraction d'année que chaque individu a effectivement passée entre
son x-ième et son (x+1)-ième anniversaire pendant la période d'observation.

### Estimateur du taux brut

À partir de l'exposition Eₓ et du nombre de décès Dₓ observés à l'âge x :

    q̂ₓ = Dₓ / Eₓ

Cet estimateur est non paramétrique et asymptotiquement sans biais sous l'hypothèse de
censure non informative. Sa variance asymptotique (approximation Poisson) est :

    Var(q̂ₓ) ≈ q̂ₓ / Eₓ  =  Dₓ / Eₓ²

L'intervalle de confiance à 95 % est alors :

    IC₉₅%(qₓ) = [ χ²(α/2 ; 2Dₓ) / (2Eₓ)  ,  χ²(1−α/2 ; 2Dₓ+2) / (2Eₓ) ]

où χ²(p ; k) désigne le quantile d'ordre p de la loi du chi-deux à k degrés de liberté.
Cette formulation (Byar, 1979) est préférable à l'approximation normale pour les petits
effectifs (Dₓ < 30).

### Lissage Whittaker-Henderson (si applicable)

Les taux bruts q̂ₓ sont lissés en minimisant le critère :

    F(λ) = Σₓ wₓ(q̂ₓ − qₓ)² + λ·Σₓ(Δ²qₓ)²

où wₓ = Eₓ est le poids de l'âge x, Δ²qₓ = qₓ − 2qₓ₊₁ + qₓ₊₂ est l'opérateur de différence
seconde, et λ est le paramètre de lissage (plus λ est grand, plus la courbe est lisse).

### Propriétés et conditions d'application

- **Non paramétrique** : aucune hypothèse sur la forme de la courbe de survie.
- **Censures à droite** : individus sortis vivants avant la fin d'observation (traités comme
  censurés à leur date de sortie).
- **Troncatures à gauche** : individus entrés en cours d'observation (date d'entrée > début
  de la période).
- **Limite** : nécessite un effectif suffisant par âge (règle pratique : Eₓ ≥ 10 pour que
  l'approximation normale soit valide ; en dessous, utiliser les IC Poisson exacts ci-dessus).

### Interprétation dans le rapport
"La durée totale d'exposition sur la période d'observation s'élève à **{E_total:.0f}
années-personnes**, répartis sur {n_ages} âges distincts. Les taux bruts q̂ₓ ont été
estimés pour les âges où l'exposition est jugée suffisante (Eₓ ≥ 10 années-personnes)."
