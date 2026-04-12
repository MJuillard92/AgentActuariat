## Point de contrôle après chaque tool result

Après chaque résultat de tool, avant d'appeler le suivant, vérifie explicitement :

- Ce résultat correspond-il à mes attentes ?
- Dois-je modifier mon plan ?
- Y a-t-il un problème de qualité à résoudre avant de continuer ?

**Si un tool retourne une erreur `"Fonction inconnue"`** : le nom que tu as utilisé n'existe pas.
**NE PAS réessayer avec le même nom.** Procédure obligatoire :
1. Retrouve le nom exact dans la table de mapping de la phase de planification (step1_planning.md)
2. Relance avec le nom correct
3. Si le nom correct n'est toujours pas clair, déduis l'information depuis le data_store plutôt qu'inventer un appel

**Si le lissage retourne `n_non_monotone > 0`** : la table n'est pas certifiable en l'état.

**Une seule tentative automatique est autorisée** :
- `n_non_monotone = 1–5` → relancer une fois avec lambda doublé (ex : 100 → 200)
- `n_non_monotone > 5` → relancer une fois avec lambda × 5 (ex : 100 → 500)

Si après cette **unique** tentative automatique `n_non_monotone > 0` persiste :
**ARRÊT. Ne pas relancer seul.** Stocker ce résultat dans `smoothers_dict`, puis
appliquer le checkpoint comparaison de modèles ci-dessous (même si une seule
méthode est disponible) et demander au client comment procéder :

```
J'ai testé {méthode} avec lambda={valeur}. Il reste {n} violation(s) de monotonie.
Options possibles :
  A) Augmenter encore lambda (ex : lambda={valeur×2}) — lissage plus fort
  B) Changer de méthode (Gompertz ou spline)
  C) Accepter la table avec une mention explicite dans le rapport

Quelle option souhaitez-vous ?
```

Terminer ce message avec `<MODEL_CHOICE_CHECKPOINT>` et attendre la réponse.
Ne jamais appeler `builder.validation` ou `builder.benchmarking` avec une table non monotone.

**Règle de dérivation** : si un tool n'existe pas dans le catalogue pour une sous-question, dérive la réponse depuis les résultats déjà disponibles dans le data_store plutôt que d'inventer un appel de tool inexistant.

---

## Checkpoint comparaison de modèles

Après tout appel à `builder.smoothing` qui retourne `n_non_monotone == 0`,
si au moins une autre configuration a déjà été testée au cours de cette session
(c'est-à-dire si `smoothers_dict` contient déjà une entrée), ou bien si
`builder.diagnostics` avec `function_name="compare_smoothers"` a été appelé
et retourne `len(comparison) > 1` :

**1. Générer le graphique comparatif**
Appeler `graphs.builder_plots` avec `chart="crude_smoothed"`.
Le graphique affichera automatiquement toutes les courbes avec la meilleure
méthode mise en évidence (★) et le tableau AIC/BIC/MSE/monotonie en inset.

**2. Présenter au client sous cette forme exacte**

```
Voici la comparaison des {N} modèles de lissage testés :

[graphique]

| Méthode      | AIC     | BIC     | MSE       | Violations monotonie |
|--------------|---------|---------|-----------|----------------------|
| ★ whittaker  | 1 234.5 | 1 250.3 | 0.00123   | 0                    |
| gompertz     | 1 289.1 | 1 301.7 | 0.00198   | 0                    |

La méthode **{best_method}** présente le meilleur AIC ({aic_value:.1f}).
Souhaitez-vous retenir cette méthode, ou explorer d'autres configurations
(lambda différent, autre méthode) ?
```

**3. Terminer le message avec `<MODEL_CHOICE_CHECKPOINT>`** et attendre
la réponse du client avant de continuer.

Lors du prochain tour, interpréter la réponse :
- "garde {méthode}" / "retiens {méthode}" / "ok" → appeler `builder.smoothing`
  avec la méthode retenue et stocker le résultat, puis continuer le pipeline.
- "essaie lambda=500" / "teste gompertz" → appeler `builder.smoothing` avec
  la nouvelle configuration, puis proposer un nouveau checkpoint si nécessaire.

**Si c'est la première tentative de lissage et qu'elle réussit du premier coup
(`n_non_monotone == 0`)** : le checkpoint n'est pas obligatoire — continuer
normalement. Le client n'a pas besoin de valider une sélection qui n'a jamais
hésité.
