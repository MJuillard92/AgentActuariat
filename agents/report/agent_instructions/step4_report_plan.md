## Rapport de certification — plan OBLIGATOIRE avant génération

Avant d'appeler `build_pdf.certification_report` ou `build_pdf.descriptive_report`,
suivre **impérativement** ces 3 étapes dans l'ordre.

---

### Étape 1 — Construire un plan dynamique depuis le data_store

Ne pas proposer un plan générique. Lire le data_store et construire le plan
en fonction de ce qui est réellement disponible :

| Clé data_store        | Section incluse dans le rapport        |
|-----------------------|----------------------------------------|
| exposure_table        | §1 Pipeline + §2 Table complète (REQUIS) |
| smoothed_table        | §2 Taux lissés + graphique taux log    |
| diagnostics           | §3 Crédibilité + graphique exposition  |
| validation            | §4 Intervalles de confiance            |
| benchmarking          | §5 Abattements + SMR global            |

Pour chaque section absente (clé manquante), la mentionner explicitement :
> "⚠ La section Validation ne sera pas incluse (builder.validation non exécuté)"

Présenter ce plan au client :

```
Voici le plan du rapport, basé sur les calculs disponibles :

**Titre** : "[titre proposé selon le portefeuille]"
**Portefeuille** : [portfolio_info résumé]
**Sexe** : [H/F]

Sections incluses :
✓ §1 Pipeline de calcul (méthode : [méthode lissage], référence : [table ref])
✓ §2 Table complète — [N] âges de [age_min] à [age_max]
✓ §3 Diagnostics — [N]% âges peu crédibles
[✓ ou ⚠ selon disponibilité] §4 Validation statistique
[✓ ou ⚠ selon disponibilité] §5 Abattements vs [référence], SMR global : [valeur]

Graphiques prévus :
- [lister uniquement les graphiques pour lesquels les données sont disponibles]

Souhaitez-vous modifier le titre, ajouter des instructions particulières,
ou puis-je générer avec ces paramètres ?
```

---

### Étape 2 — Rédiger et soumettre le commentary pour validation

Une fois le plan approuvé par le client, rédiger les §1 à §5 du commentary
(voir behavioral_contract.md — Étapes A, B, C).

**Puis présenter un résumé des §1-§5 au client avant de générer le PDF :**

```
Voici le contenu narratif que j'ai rédigé pour le rapport.
Confirmez-vous que je peux générer le PDF avec ce contenu ?

§1 Contexte : [2-3 phrases résumant le §1 rédigé]
§2 Méthode : [1-2 phrases sur le choix de lissage et sa justification]
§3 Résultats : [2-3 phrases sur SMR, anomalies détectées, déciles]
§4 Limites : [1-2 phrases sur âges peu crédibles, exclusions]
§5 Conclusion : [1-2 phrases sur l'utilisabilité recommandée]

[Confirmer / Modifier]
```

Ne pas appeler le tool tant que le client n'a pas confirmé ce résumé
ou dit "oui", "génère", "vas-y".

**Exception** : si le client dit "génère directement" dès le départ,
sauter l'étape 2 et appeler le tool avec le commentary complet sans
présenter le résumé intermédiaire.

---

### Étape 3 — Appeler le tool avec les paramètres complets

Injecter dans les params **exactement** ce que le plan et le commentary
contiennent — pas de valeurs par défaut silencieuses si le client a
fourni des instructions :

- `params.title` : titre validé à l'étape 1
- `params.portfolio_info` : description du portefeuille
- `params.sexe` : "H" ou "F"
- `params.commentary` : texte complet §1-§5 rédigé à l'étape 2
  (800-1200 mots, paragraphes séparés par `\n\n`, sans markdown)

**Règle de cohérence** : si le plan de l'étape 1 indiquait qu'une section
serait présente (ex : §5 Abattements), vérifier que la clé correspondante
est dans le data_store avant d'appeler le tool. Si elle a disparu (ex :
l'utilisateur a réinitialisé), informer le client et mettre à jour le plan
avant de générer.
